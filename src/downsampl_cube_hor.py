import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

def downsample_cube_horizontally(cube, read=False, output_path=None, downsampling_factor=2,
                                 method='decimate', apply_antialias=False):
    """
    Downsample seismic cube horizaontlly (xy traces) choosing on of the strategy.
    
    Parameters:
    -----------
    cube : str | xr.ds
        Path to input seismic cube (.nc file)
    output_path : str or None
        Path to save downsampled cube. If None, returns xarray Dataset
    downsampling_factor : int
        Downsampling factor (e.g., 2 = take every 2nd inline/xline)
    method : str
        'decimate' - simple decimation (take every nth sample)
        'average' - block averaging
        'max' - take maximum in each block
        'min' - take minimum in each block
    apply_antialias : bool
        Whether to apply anti-aliasing filter before decimation
    
    Returns:
    --------
    xr.Dataset or None (if saved to file)
    """
    print(f"Downsampling factor: {downsampling_factor}")
    print(f"Method: {method}")
    if read:
        print(f"Downsampling cube: {cube}")
        
        
        # Load the cube
        print("Loading cube...")
        cube_ds = xr.open_dataset(cube)
    else:
        print('Work with preloaded cube')
        cube_ds = cube
    
    # Get data variable name (assuming first variable is seismic data)
    data_var_name = list(cube_ds.data_vars)[0]
    data_array = cube_ds[data_var_name]
    
    print(f"Original shape: {data_array.shape}")
    print(f"Original dimensions: {dict(data_array.sizes)}")
    
    # Get original coordinates
    original_iline = data_array.iline.values
    original_xline = data_array.xline.values
    original_twt = data_array.twt.values
    
    # Get CDP coordinates
    cdp_x = cube_ds.cdp_x.values
    cdp_y = cube_ds.cdp_y.values
    
    print(f"Original iline range: {original_iline.min()} to {original_iline.max()}")
    print(f"Original xline range: {original_xline.min()} to {original_xline.max()}")
    print(f"Original TWT range: {original_twt.min():.1f} to {original_twt.max():.1f} ms")
    
    # Handle different downsampling methods
    print(f"\nDownsampling data using method: '{method}'...")
    
    data = data_array.values
    n_ilines, n_xlines, n_twt = data.shape
    
    if method == 'decimate':
        # Simple decimation - fastest method
        if apply_antialias:
            print("Applying anti-aliasing filter...")
            from scipy.ndimage import uniform_filter
            data_smoothed = uniform_filter(data, size=(1, 1, 0))
            downsampled_data = data_smoothed[::downsampling_factor, ::downsampling_factor, :]
        else:
            downsampled_data = data[::downsampling_factor, ::downsampling_factor, :]
        
        # Get coordinates for decimated data
        downsampled_iline = original_iline[::downsampling_factor]
        downsampled_xline = original_xline[::downsampling_factor]
    
    elif method in ['average', 'max', 'min']:
        # Block operations - handle dimension mismatches carefully
        print(f"Performing block {method}...")
        
        # Calculate dimensions that work for block operations
        # We need dimensions that are multiples of downsampling_factor
        new_n_ilines = n_ilines // downsampling_factor
        new_n_xlines = n_xlines // downsampling_factor
        
        # Calculate actual data size we can use
        usable_ilines = new_n_ilines * downsampling_factor
        usable_xlines = new_n_xlines * downsampling_factor
        
        print(f"Original: {n_ilines} ilines, {n_xlines} xlines")
        print(f"Usable for blocks: {usable_ilines} ilines, {usable_xlines} xlines")
        print(f"Will lose: {n_ilines - usable_ilines} ilines, {n_xlines - usable_xlines} xlines")
        
        # Truncate data to usable dimensions
        truncated_data = data[:usable_ilines, :usable_xlines, :]
        
        # Reshape for block operation
        reshaped = truncated_data.reshape(
            new_n_ilines, downsampling_factor,
            new_n_xlines, downsampling_factor,
            n_twt
        )
        
        # Apply operation
        if method == 'average':
            downsampled_data = reshaped.mean(axis=(1, 3))
        elif method == 'max':
            downsampled_data = reshaped.max(axis=(1, 3))
        elif method == 'min':
            downsampled_data = reshaped.min(axis=(1, 3))
        
        # Get corresponding coordinates
        # We need to match the truncated dimensions
        downsampled_iline = original_iline[:usable_ilines:downsampling_factor]
        downsampled_xline = original_xline[:usable_xlines:downsampling_factor]
        
        print(f"Downsampled shape: {downsampled_data.shape}")
        print(f"Downsampled ilines: {len(downsampled_iline)}")
        print(f"Downsampled xlines: {len(downsampled_xline)}")
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: decimate, average, max, min")
    
    print(f"Final downsampled data shape: {downsampled_data.shape}")
    
    # Downsample CDP coordinates to match
    if cdp_x.ndim == 1:
        # 1D coordinates
        if method == 'decimate':
            downsampled_cdp_x = cdp_x[::downsampling_factor]
            downsampled_cdp_y = cdp_y[::downsampling_factor]
        else:
            # For block methods, use same pattern as data
            downsampled_cdp_x = cdp_x[:len(downsampled_iline) * downsampling_factor:downsampling_factor]
            downsampled_cdp_y = cdp_y[:len(downsampled_xline) * downsampling_factor:downsampling_factor]
    else:
        # 2D coordinates
        if method == 'decimate':
            downsampled_cdp_x = cdp_x[::downsampling_factor, ::downsampling_factor]
            downsampled_cdp_y = cdp_y[::downsampling_factor, ::downsampling_factor]
        else:
            # For block methods, truncate to match data
            usable_cdp_ilines = len(downsampled_iline) * downsampling_factor
            usable_cdp_xlines = len(downsampled_xline) * downsampling_factor
            downsampled_cdp_x = cdp_x[:usable_cdp_ilines:downsampling_factor, 
                                     :usable_cdp_xlines:downsampling_factor]
            downsampled_cdp_y = cdp_y[:usable_cdp_ilines:downsampling_factor, 
                                     :usable_cdp_xlines:downsampling_factor]
    
    # Verify all dimensions match
    assert downsampled_data.shape[0] == len(downsampled_iline), \
        f"iline mismatch: data={downsampled_data.shape[0]}, coords={len(downsampled_iline)}"
    assert downsampled_data.shape[1] == len(downsampled_xline), \
        f"xline mismatch: data={downsampled_data.shape[1]}, coords={len(downsampled_xline)}"
    assert downsampled_data.shape[2] == len(original_twt), \
        f"twt mismatch: data={downsampled_data.shape[2]}, coords={len(original_twt)}"
    
    # Create new xarray Dataset
    print("\nCreating downsampled dataset...")
    
    # Create coordinate arrays
    coords = {
        'iline': downsampled_iline,
        'xline': downsampled_xline,
        'twt': original_twt
    }
    
    # Create DataArray with downsampled data
    downsampled_da = xr.DataArray(
        data=downsampled_data,
        dims=('iline', 'xline', 'twt'),
        coords=coords,
        attrs=data_array.attrs
    )
    
    # Create Dataset
    downsampled_ds = xr.Dataset({data_var_name: downsampled_da})
    
    # Add CDP coordinates
    if cdp_x.ndim == 1:
        # 1D case
        assert len(downsampled_cdp_x) == len(downsampled_iline), \
            f"cdp_x length mismatch: {len(downsampled_cdp_x)} != {len(downsampled_iline)}"
        assert len(downsampled_cdp_y) == len(downsampled_xline), \
            f"cdp_y length mismatch: {len(downsampled_cdp_y)} != {len(downsampled_xline)}"
        
        downsampled_ds['cdp_x'] = xr.DataArray(downsampled_cdp_x, dims='iline')
        downsampled_ds['cdp_y'] = xr.DataArray(downsampled_cdp_y, dims='xline')
    else:
        # 2D case
        assert downsampled_cdp_x.shape[0] == len(downsampled_iline), \
            f"cdp_x iline mismatch: {downsampled_cdp_x.shape[0]} != {len(downsampled_iline)}"
        assert downsampled_cdp_x.shape[1] == len(downsampled_xline), \
            f"cdp_x xline mismatch: {downsampled_cdp_x.shape[1]} != {len(downsampled_xline)}"
        
        downsampled_ds['cdp_x'] = xr.DataArray(
            downsampled_cdp_x,
            dims=('iline', 'xline')
        )
        downsampled_ds['cdp_y'] = xr.DataArray(
            downsampled_cdp_y,
            dims=('iline', 'xline')
        )
    
    # Copy other variables and attributes
    for var in cube_ds.variables:
        if var not in [data_var_name, 'iline', 'xline', 'twt', 'cdp_x', 'cdp_y']:
            downsampled_ds[var] = cube_ds[var]
    
    downsampled_ds.attrs = cube_ds.attrs
    
    # Calculate compression ratio
    original_size = data_array.size * data_array.dtype.itemsize / 1e9
    downsampled_size = downsampled_data.size * downsampled_data.dtype.itemsize / 1e9
    compression_ratio = original_size / downsampled_size
    
    print(f"\nSize reduction: {original_size:.2f} GB -> {downsampled_size:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Memory savings: {(original_size - downsampled_size):.2f} GB")
    
    # Save or return
    if output_path:
        print(f"\nSaving to: {output_path}")
        
        # Set encoding for efficient storage
        encoding = {
            data_var_name: {
                'zlib': True,
                'complevel': 1,
                'dtype': 'float32'
            }
        }
        
        downsampled_ds.to_netcdf(output_path, encoding=encoding)
        print("Save complete!")
        cube_ds.close()
        return None
    else:
        cube_ds.close()
        return downsampled_ds

def get_cube_info(cube_path):
    """
    Get information about cube dimensions.
    """
    print(f"\nAnalyzing cube: {cube_path}")
    
    cube_ds = xr.open_dataset(cube_path)
    data_var = list(cube_ds.data_vars)[0]
    data_array = cube_ds[data_var]
    
    n_ilines, n_xlines, n_twt = data_array.shape
    
    print(f"Dimensions: {n_ilines} ilines × {n_xlines} xlines × {n_twt} twt")
    print(f"Total elements: {n_ilines * n_xlines * n_twt:,}")
    
    # Check if dimensions are divisible by 2
    print(f"\nDivisibility check for factor 2:")
    print(f"  ILINES: {n_ilines} ÷ 2 = {n_ilines / 2} (remainder: {n_ilines % 2})")
    print(f"  XLINE: {n_xlines} ÷ 2 = {n_xlines / 2} (remainder: {n_xlines % 2})")
    
    if n_ilines % 2 != 0 or n_xlines % 2 != 0:
        print(f"  WARNING: Dimensions not divisible by 2!")
        print(f"  Block methods (average/max/min) will lose {n_ilines % 2} ilines, {n_xlines % 2} xlines")
    
    cube_ds.close()
    
    return n_ilines, n_xlines, n_twt

# Example usage
# if __name__ == "__main__":
#     cube_path = "your_seismic_cube.nc"
    
#     # First, check cube dimensions
#     n_ilines, n_xlines, n_twt = get_cube_info(cube_path)
    
#     # Method 1: Simple decimation (works with any dimensions)
#     print("\n" + "="*60)
#     print("Method 1: Simple decimation")
#     downsample_cube_horizontally(
#         cube_path=cube_path,
        #   read=True,
#         output_path="cube_decimated.nc",
#         downsampling_factor=2,
#         method='decimate'
#     )
    
#     # Method 2: Block averaging
#     print("\n" + "="*60)
#     print("Method 2: Block averaging")
    
#     if n_ilines % 2 == 0 and n_xlines % 2 == 0:
#         downsample_cube_horizontally(
#             cube=cube_path,
#             read=True,
#             output_path="cube_averaged.nc",
#             downsampling_factor=2,
#             method='average'
#         )
#     else:
#         print(f"Cannot use 'average' method: dimensions not divisible by 2")
#         print(f"ILINES: {n_ilines} (remainder: {n_ilines % 2})")
#         print(f"XLINE: {n_xlines} (remainder: {n_xlines % 2})")
#         print("Use 'decimate' method instead.")

# for cube in os.listdir(cubes_fold):
#     cube_path = os.path.join(cubes_fold, cube)
#     down_cube = downsample_cube_horizontally(
#                             cube=cube_path,
#                               read=True,
#                             output_path=None,
#                             downsampling_factor=2,
#                             method='average'
#                         )
#     save_path = os.path.join(save_fold, 
#                             f'{cube.split(".")[0]}_down_horzn.nc')
#     down_cube.to_netcdf(save_path, engine='h5netcdf')