import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from typing import Tuple, List, Optional, Union, Dict
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class SeismicMapExtractor:
    """
    Extract seismic data at surface points and create 2D maps in flattened plane.
    
    Parameters:
    -----------
    max_distance : float
        Maximum allowed distance in meters (points beyond this will be masked)
    use_dask : bool
        Whether to use dask for parallel processing
    chunks : dict or 'auto'
        Chunk specification for dask arrays
    """
    
    def __init__(self, max_distance: float = 100.0, use_dask: bool = True, 
                 chunks: Union[dict, str] = 'auto'):
        self.max_distance = max_distance
        self.use_dask = use_dask
        self.chunks = chunks
        self.index_map = None
        self.surface_points = None
        self.iline_indices = None
        self.xline_indices = None
        self.distances = None
        self.valid_mask = None
        self.extracted_cube_cdps = None  # Store actual cube CDP coordinates at extracted points
        self.curve = None

    def build_index_map(self, 
                       reference_cube: xr.Dataset,
                       surface_df: pd.DataFrame,
                       x_coord: str = 'cdp_x',
                       y_coord: str = 'cdp_y',
                       twt_coord: str = 'twt') -> None:
        """
        Build persistent index map from surface points to cube indices.
        Also stores the actual cube CDP coordinates for each surface point.
        
        Parameters:
        -----------
        reference_cube : xr.Dataset
            Reference cube containing CDP coordinates
        surface_df : pd.DataFrame
            DataFrame with surface points containing cdp_x, cdp_y columns
        x_coord, y_coord : str
            Names of coordinate variables in the cube
        twt_coord : str
            Name of the time/depth coordinate
        """
        print("=" * 60)
        print("BUILDING SPATIAL INDEX MAP")
        print("=" * 60)
        
        # Extract CDP coordinates from the reference cube
        cdp_x = reference_cube[x_coord].values #2d
        cdp_y = reference_cube[y_coord].values #2d
        twt = reference_cube[twt_coord].values #1d
        
        # Get the shape and create flattened array of coordinates
        n_iline, n_xline = cdp_x.shape
        n_twt = len(twt)
        
        print(f"Cube dimensions: iline={n_iline}, xline={n_xline}, twt={n_twt}")
        print(f"Number of surface points: {len(surface_df)}")
        
        # Create mesh of iline and xline indices
        iline_indices, xline_indices = np.meshgrid(
            np.arange(n_iline), 
            np.arange(n_xline), 
            indexing='ij'
        )
        
        # Flatten everything for KD-tree
        points = np.column_stack([
            cdp_x.ravel(),
            cdp_y.ravel()
        ])
        
        # Remove any NaN points
        valid_mask = ~(np.isnan(points).any(axis=1))
        points = points[valid_mask]
        flat_iline = iline_indices.ravel()[valid_mask]
        flat_xline = xline_indices.ravel()[valid_mask]
        
        # Build KD-tree
        print("Building KD-tree...")
        self.tree = cKDTree(points)
        
        # Store grid information for later use
        self.grid_shape = (n_iline, n_xline)
        self.flat_iline = flat_iline
        self.flat_xline = flat_xline
        self.cube_cdp_x = cdp_x
        self.cube_cdp_y = cdp_y
        self.twt_coords = twt
        
        # Query for all surface points
        surface_coords = surface_df[['cdp_x', 'cdp_y']].values
        
        print(f"Querying KD-tree for nearest neighbors...")
        distances, indices = self.tree.query(surface_coords)
        
        # Map back to original indices
        self.iline_indices = flat_iline[indices]
        self.xline_indices = flat_xline[indices]
        self.distances = distances
        
        # Apply distance threshold
        self.valid_mask = distances <= self.max_distance
        
        # Store surface points for reference
        self.surface_points = surface_df.copy()
        
        # Extract the ACTUAL cube CDP coordinates at the extracted points
        self.extracted_cube_cdps = pd.DataFrame({
            'surface_index': np.arange(len(surface_df)),
            'cube_cdp_x': cdp_x[self.iline_indices, self.xline_indices],
            'cube_cdp_y': cdp_y[self.iline_indices, self.xline_indices],
            'distance': distances,
            'valid': self.valid_mask
        })
        
        # Find nearest twt indices for each surface point
        if 'twt_near' in surface_df.columns:
            twt_values = surface_df['twt_near'].values
            self.twt_indices = np.abs(twt[:, np.newaxis] - twt_values).argmin(axis=0)
            self.twt_values_actual = twt[self.twt_indices]
        else:
            self.twt_indices = None
            self.twt_values_actual = None
        
        # Report statistics
        n_valid = self.valid_mask.sum()
        print("\n" + "=" * 60)
        print("INDEX MAP BUILD COMPLETE")
        print("=" * 60)
        print(f"Valid points within {self.max_distance}m: {n_valid} / {len(surface_df)} ({100*n_valid/len(surface_df):.1f}%)")
        print(f"Mean distance: {distances.mean():.2f}m")
        print(f"Max distance: {distances.max():.2f}m")
        print(f"Std distance: {distances.std():.2f}m")
        
        # Show distance distribution
        percentiles = [10, 25, 50, 75, 90, 95]
        print("\nDistance percentiles:")
        for p in percentiles:
            print(f"  {p}th: {np.percentile(distances, p):.2f}m")
        
    def save_index_map(self, filename: str):
        """Save index map to numpy compressed file."""
        np.savez_compressed(
            filename,
            iline_indices=self.iline_indices,
            xline_indices=self.xline_indices,
            distances=self.distances,
            valid_mask=self.valid_mask,
            grid_shape=self.grid_shape,
            max_distance=self.max_distance,
            twt_indices=self.twt_indices,
            twt_values_actual=self.twt_values_actual
        )
        print(f"Index map saved to {filename}")
        
    def load_index_map(self, filename: str):
        """Load pre-computed index map."""
        data = np.load(filename, allow_pickle=True)
        self.iline_indices = data['iline_indices']
        self.xline_indices = data['xline_indices']
        self.distances = data['distances']
        self.valid_mask = data['valid_mask']
        self.grid_shape = tuple(data['grid_shape'])
        self.max_distance = data['max_distance']
        
        # Load optional TWT data if available
        if 'twt_indices' in data:
            self.twt_indices = data['twt_indices']
            self.twt_values_actual = data['twt_values_actual']
        else:
            self.twt_indices = None
            self.twt_values_actual = None
            
        print(f"Index map loaded from {filename}")
        print(f"Valid points: {self.valid_mask.sum()} / {len(self.valid_mask)} ({100*self.valid_mask.sum()/len(self.valid_mask):.1f}%)")
    
    def set_surface_as_df(self, surface_points):
        '''set preloaded surace points. it is the same regardless the resolution'''

        self.surface_points = surface_points

    def read_surface_points(self, df_surf_path):
        '''surace of points is the same regardless the resolution'''
        
        self.surface_points = pd.read_csv(df_surf_path)
        

    def extract_and_create_map(self,
                          cube: xr.Dataset,
                          data_var: str = 'data',
                          output_nc_file: Optional[str] = None,
                          x_coord: str = 'cdp_x',
                          y_coord: str = 'cdp_y',
                          sec_name=None, sec_val=None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract values from cube and create a 2D map on the original CDP grid.
        Points without data are set to NaN (will appear as white in plots).
        
        Parameters:
        -----------
        cube : xr.Dataset
            Cube to extract values from
        data_var : str
            Variable name to extract
        output_nc_file : str, optional
            If provided, save results to NetCDF
        x_coord, y_coord : str
            Names of coordinate variables in the cube
            
        Returns:
        --------
        Tuple containing:
        - DataFrame with all extracted information
        - 2D array of X coordinates (full CDP grid from cube)
        - 2D array of Y coordinates (full CDP grid from cube)
        - 2D array of data values (NaN for points not in surface)
        """
        if self.iline_indices is None:
            raise ValueError("Index map not built or loaded. Call build_index_map() or load_index_map() first.")
        
        print(f"\nExtracting {data_var} from cube...")
        
        # Get the full CDP grid from the cube
        cdp_x_full = cube[x_coord].values
        cdp_y_full = cube[y_coord].values
        
        # Prepare results DataFrame
        results = self.surface_points.copy()
        results['distance_to_cube'] = self.distances
        results['valid_point'] = self.valid_mask
        results['iline_index'] = self.iline_indices
        results['xline_index'] = self.xline_indices
        
        # Get indices for valid points only
        valid_iline = self.iline_indices[self.valid_mask]
        valid_xline = self.xline_indices[self.valid_mask]
        
        # Initialize data array with NaN
        data_values = np.full(len(self.valid_mask), np.nan)
        
        if (sec_name is not None) and (sec_val is not None):
            section = cube.sel({sec_name: sec_val}, method='nearest')
            self.curve = pd.DataFrame({x_coord: section[x_coord].values,
                                       y_coord: section[y_coord].values})

        if len(valid_iline) > 0:
            # Extract data from cube
            data = cube[data_var]
            
            # Handle 3D vs 2D data
            if 'twt' in data.dims and self.twt_indices is not None:
                # 3D data - extract at specific TWT
                valid_twt = self.twt_indices[self.valid_mask]
                
                # Use vectorized indexing for performance
                extracted = data.isel(
                    twt=xr.DataArray(valid_twt, dims="points"),
                    iline=xr.DataArray(valid_iline, dims="points"),
                    xline=xr.DataArray(valid_xline, dims="points")
                ).values
            else:
                # 2D data
                extracted = data.isel(
                    iline=xr.DataArray(valid_iline, dims="points"),
                    xline=xr.DataArray(valid_xline, dims="points")
                ).values
            
            data_values[self.valid_mask] = extracted
        
        results[data_var] = data_values
        
        # Save to NetCDF if requested
        if output_nc_file:
            self._save_to_netcdf(results, output_nc_file)
        
        # Create 2D map on the original CDP grid
        print("Creating 2D map on original CDP grid...")
        
        # Initialize the full grid with NaN
        Z_map = np.full(cdp_x_full.shape, np.nan)
        
        # Place extracted values at their positions
        Z_map[valid_iline, valid_xline] = data_values[self.valid_mask]
        
        # Use the full CDP grids for X and Y
        X_map = cdp_x_full
        Y_map = cdp_y_full
        
        # Report statistics
        n_valid_cells = np.sum(~np.isnan(Z_map))
        print(f"\nMap Statistics:")
        print(f"  Original CDP grid shape: {Z_map.shape}")
        print(f"  Total CDP points in cube: {Z_map.size}")
        print(f"  Surface points intersecting cube: {n_valid_cells} ({100*n_valid_cells/Z_map.size:.2f}%)")
        if n_valid_cells > 0:
            print(f"  Data range: [{np.nanmin(Z_map):.2f}, {np.nanmax(Z_map):.2f}]")
            print(f"  Data mean ± std: {np.nanmean(Z_map):.2f} ± {np.nanstd(Z_map):.2f}")
        
        return results, X_map, Y_map, Z_map

    def plot_map_on_grid(self,
                        X: np.ndarray,
                        Y: np.ndarray,
                        Z: np.ndarray,
                        title: str = "Seismic Attribute Map",
                        figsize: Tuple[int, int] = (14, 10),
                        cmap: str = 'RdBu_r',
                        save_path: Optional[str] = None,
                        show_colorbar: bool = True,
                        plot_all_cdp: bool = True,
                        marker_size: float = 5,
                        **scatter_kwargs):
        """
        Plot the 2D map on the original CDP grid.
        Points not in the surface appear as blank/white.
        
        Parameters:
        -----------
        X, Y : 2D arrays
            Full CDP coordinate grids from cube
        Z : 2D array
            Data values on CDP grid (NaN for points not in surface)
        title : str
            Plot title
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        show_colorbar : bool
            Whether to show colorbar
        plot_all_cdp : bool
            If True, plot all CDP positions as light gray dots. If False, plot only surface points.
        marker_size : float
            Size of markers for scatter plot
        **scatter_kwargs : additional arguments for scatter plot
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Flatten the arrays for scatter plotting
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        
        # Create mask for valid (non-NaN) points
        valid_mask = ~np.isnan(Z_flat)
        
        # Plot all CDP positions as background if requested
        if plot_all_cdp:
            ax.scatter(
                X_flat, Y_flat,
                c='lightgray',
                s=marker_size * 0.5,
                alpha=0.3,
                label='All CDP positions',
                **({})
            )
        
        # Plot the valid surface points with color mapping
        if valid_mask.any():
            scatter = ax.scatter(
                X_flat[valid_mask], 
                Y_flat[valid_mask],
                c=Z_flat[valid_mask],
                s=marker_size * 2,
                cmap=cmap,
                alpha=0.8,
                edgecolors='none',
                label='Surface points',
                **({}),
                **scatter_kwargs
            )
            
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, label='Amplitude', extend='both')
        else:
            print("Warning: No valid data points to plot")
        
        
        ax.set_xlabel('CDP X (m)')
        ax.set_ylabel('CDP Y (m)')
        ax.set_aspect('equal')
    
        # Add statistics to title
        n_valid = valid_mask.sum()
        total = Z.size
        if n_valid > 0:
            title = (f"{title}\n"
                    f"Surface points: {n_valid} / {total} ({100*n_valid/total:.1f}% of CDP grid) | "
                    f"Range: [{np.nanmin(Z):.2f}, {np.nanmax(Z):.2f}]")
        else:
            title = f"{title}\nNo valid surface points in this cube"
        
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig, ax

    def plot_map_as_image(self,
                        X: np.ndarray,
                        Y: np.ndarray,
                        Z: np.ndarray,
                        title: str = "Seismic Attribute Map",
                        figsize: Tuple[int, int] = (14, 10),
                        cmap: str = 'RdBu_r',
                        save_path: Optional[str] = None,
                        show_colorbar: bool = True,
                        interpolation: str = 'none',
                        **imshow_kwargs):
        """
        Plot the map as an image (pcolormesh) on the original grid.
        Useful when the grid is regular and you want filled cells.
        
        Parameters:
        -----------
        X, Y : 2D arrays
            Full CDP coordinate grids from cube
        Z : 2D array
            Data values on CDP grid (NaN for points not in surface)
        title : str
            Plot title
        figsize : tuple
            Figure size
        cmap : str
            Colormap name
        save_path : str, optional
            Path to save the figure
        show_colorbar : bool
            Whether to show colorbar
        interpolation : str
            Interpolation method for imshow (if using regular grid)
        **imshow_kwargs : additional arguments for pcolormesh/imshow
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a masked array for NaN values (they will appear white)
        Z_masked = np.ma.masked_where(np.isnan(Z), Z)
        
        # Default plot parameters
        plot_kwargs = {
            'cmap': cmap,
            'shading': 'auto',
            **imshow_kwargs
        }
        
        # Check if grid is regular enough for imshow
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        
        if len(x_unique) * len(y_unique) == X.size:
            # Regular grid - can use imshow for better performance
            x_sorted = np.sort(x_unique)
            y_sorted = np.sort(y_unique)[::-1]  # Reverse for correct orientation
            
            # Reorder Z to match sorted coordinates
            Z_reordered = np.zeros((len(y_sorted), len(x_sorted)))
            for i, y_val in enumerate(y_sorted):
                for j, x_val in enumerate(x_sorted):
                    mask = (X == x_val) & (Y == y_val)
                    if mask.any():
                        Z_reordered[i, j] = Z[mask][0]
                    else:
                        Z_reordered[i, j] = np.nan
            
            Z_masked_reordered = np.ma.masked_where(np.isnan(Z_reordered), Z_reordered)
            
            im = ax.imshow(Z_masked_reordered, 
                        extent=[x_sorted.min(), x_sorted.max(), 
                                y_sorted.min(), y_sorted.max()],
                        origin='upper',
                        interpolation=interpolation,
                        **plot_kwargs)
            ax.set_xlabel('CDP X (m)')
            ax.set_ylabel('CDP Y (m)')
        else:
            # Irregular grid - use pcolormesh
            im = ax.pcolormesh(X, Y, Z_masked, **plot_kwargs)
            ax.set_xlabel('CDP X (m)')
            ax.set_ylabel('CDP Y (m)')
        
        ax.set_aspect('equal')
        
        # Add colorbar if requested
        if show_colorbar and not np.all(np.isnan(Z)):
            plt.colorbar(im, ax=ax, label='Amplitude', extend='both')
        
        # Add statistics to title
        n_valid = np.sum(~np.isnan(Z))
        total = Z.size
        if n_valid > 0:
            title = (f"{title}\n"
                    f"Surface points: {n_valid} / {total} ({100*n_valid/total:.1f}% of CDP grid) | "
                    f"Range: [{np.nanmin(Z):.2f}, {np.nanmax(Z):.2f}]")
        else:
            title = f"{title}\nNo valid surface points in this cube"
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # plt.show()
        
        return fig, ax
    
        
        
        # self.iline_indices 
        # self.xline_indices
    
    def plot_cluster_map_as_image(self,
                             X: np.ndarray,
                             Y: np.ndarray,
                             Z: np.ndarray,
                             title: str = "Cluster Map",
                             figsize: Tuple[int, int] = (14, 10),
                             cmap: str = 'tab10',  # Categorical colormap
                             show_colorbar: bool = True,
                             interpolation: str = 'none',
                             class_names: Optional[Dict[int, str]] = None,
                             **imshow_kwargs):
        """
        Plot a cluster/classification map as an image on the original grid.
        Designed for discrete class values (integers). Colorbar only shows classes that exist in data.
        
        Parameters:
        -----------
        X, Y : 2D arrays
            Full CDP coordinate grids from cube
        Z : 2D array
            Cluster/class values on CDP grid (NaN or -1 for points not in surface)
            Should contain integer values representing different clusters
        title : str
            Plot title
        figsize : tuple
            Figure size
        cmap : str
            Colormap name - use categorical colormaps like 'tab10', 'tab20', 'Set3'
        show_colorbar : bool
            Whether to show colorbar
        interpolation : str
            Interpolation method for imshow (should be 'none' for discrete data)
        class_names : dict, optional
            Dictionary mapping class values to names for colorbar labels
        nan_color : str
            Color for NaN/empty cells (not used directly - NaN appears transparent/white)
        **imshow_kwargs : additional arguments for imshow/pcolormesh
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle -1 values as NaN (commonly used for no-data in classification)
        Z_clean = Z.copy()
        if -1 in Z_clean:
            Z_clean[Z_clean == -1] = np.nan
        
        # Get unique classes (excluding NaN) and sort them
        unique_classes = np.unique(Z_clean[~np.isnan(Z_clean)])
        unique_classes = np.sort(unique_classes)  # Ensure consistent ordering
        n_classes = len(unique_classes)
        
        if n_classes == 0:
            print("Warning: No valid class data to plot")
            ax.set_title(f"{title}\nNo valid class data")
            ax.set_aspect('equal')
            plt.tight_layout()
            return fig, ax
        
        print(f"Plotting cluster map with {n_classes} classes: {unique_classes}")
        
        # APPROACH 1: Create mapping from original class values to consecutive indices
        # Create a dictionary mapping original class values to new consecutive indices (0, 1, 2, ...)
        class_to_idx = {orig_val: idx for idx, orig_val in enumerate(unique_classes)}
        idx_to_class = {idx: orig_val for idx, orig_val in enumerate(unique_classes)}
        
        # Create a new array with mapped indices for plotting
        Z_mapped = np.full_like(Z_clean, np.nan, dtype=float)
        for orig_val in unique_classes:
            mask = (Z_clean == orig_val)
            Z_mapped[mask] = class_to_idx[orig_val]
        
        # Create discrete colormap with exactly n_classes colors
        from matplotlib.colors import ListedColormap, BoundaryNorm
        
        # Get the base colormap and extract colors
        base_cmap = plt.get_cmap(cmap)
        
        if hasattr(base_cmap, 'colors') and len(base_cmap.colors) >= n_classes:
            # If colormap has discrete colors, use them directly
            colors = base_cmap.colors[:n_classes]
        else:
            # Sample colors from continuous colormap
            colors = [base_cmap(i / n_classes) for i in range(n_classes)]
        
        # Create discrete colormap
        discrete_cmap = ListedColormap(colors)
        
        # Create normalization for the mapped indices (0 to n_classes-1)
        bounds = np.arange(n_classes + 1) - 0.5
        norm = BoundaryNorm(bounds, n_classes)
        
        plot_kwargs = {
            'cmap': discrete_cmap,
            'norm': norm,
            **imshow_kwargs
        }
        
        # Create a masked array for NaN values
        Z_masked = np.ma.masked_where(np.isnan(Z_mapped), Z_mapped)
        
        # Check if grid is regular enough for imshow
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        
        if len(x_unique) * len(y_unique) == X.size:
            print('use imshow')
            # Regular grid - use imshow
            x_sorted = np.sort(x_unique)
            y_sorted = np.sort(y_unique)[::-1]  # Reverse for correct orientation
            
            # Reorder Z to match sorted coordinates
            Z_reordered = np.full((len(y_sorted), len(x_sorted)), np.nan)
            
            # Create mapping for faster lookup
            x_to_idx = {x: i for i, x in enumerate(x_sorted)}
            y_to_idx = {y: i for i, y in enumerate(y_sorted)}
            
            # Flatten arrays for vectorized assignment
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z_mapped.flatten()
            
            for idx in range(len(X_flat)):
                if not np.isnan(Z_flat[idx]):
                    i = y_to_idx[Y_flat[idx]]
                    j = x_to_idx[X_flat[idx]]
                    Z_reordered[i, j] = Z_flat[idx]
            
            Z_masked_reordered = np.ma.masked_where(np.isnan(Z_reordered), Z_reordered)
            
            im = ax.imshow(Z_masked_reordered, 
                        extent=[x_sorted.min(), x_sorted.max(), 
                                y_sorted.min(), y_sorted.max()],
                        origin='upper',
                        interpolation=interpolation, rasterized=True,
                        **plot_kwargs)
            ax.set_xlabel('CDP X (m)')
            ax.set_ylabel('CDP Y (m)')
        else:
            # Irregular grid - use pcolormesh
            im = ax.pcolormesh(X, Y, Z_masked, shading='auto', rasterized=True,
                                 **plot_kwargs)
            ax.set_xlabel('CDP X (m)')
            ax.set_ylabel('CDP Y (m)')
        if self.curve is not None:
            #plot cross_section
            ax.plot(self.curve['cdp_x'], self.curve['cdp_y'], color='black', linewidth=3, linestyle='--')
        ax.set_aspect('equal')
        
        # Add colorbar if requested - now showing only classes that exist
        if show_colorbar:
            # Create colorbar
            cbar = plt.colorbar(im, ax=ax, extend='neither')
            
            # Set ticks at the mapped indices (0, 1, 2, ...)
            tick_positions = np.arange(n_classes)
            cbar.set_ticks(tick_positions)
            
            # Create tick labels using original class values and optional names
            if class_names:
                # Use provided class names
                tick_labels = [class_names.get(idx_to_class[i], f'Class {int(idx_to_class[i])}') 
                            for i in range(n_classes)]
            else:
                # Use original class values
                tick_labels = [f'{int(idx_to_class[i])}' for i in range(n_classes)]
            
            cbar.set_ticklabels(tick_labels)
            cbar.set_label('Cluster/Class')
        # Add statistics to title
        n_valid = np.sum(~np.isnan(Z_clean))
        total = Z.size
        # self
        if n_valid > 0:
            # Calculate class distribution
            # class_distribution = []
            # for orig_val in unique_classes:
            #     count = np.sum(Z_clean == orig_val)
            #     if class_names and orig_val in class_names:
            #         class_distribution.append(f"{class_names[orig_val]}: {count}")
            #     else:
            #         class_distribution.append(f"Class {int(orig_val)}: {count}")
            
            # class_distr_str = " | ".join(class_distribution)
            
            title = (f"{title}\n")
        else:
            title = f"{title}\nNo valid class data in this cube"
        
        ax.set_title(title, fontsize=15)
        
        plt.tight_layout()
        
        return fig, ax
