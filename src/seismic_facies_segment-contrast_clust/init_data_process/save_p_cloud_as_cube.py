
import xarray as xr
from segysak.segy import segy_loader
from segysak.segy import segy_writer
import xarray as xr
import numpy as np
import os

class SavePredictionPCloud:

    """Read point cloud as separate numpy arrays with dims (iline, xline, twt)
       and property to write to save it either as xarray cube or .segy file for Petrel
       save_fold - fold to save the composed file;
       bolvanka_path - path to a prepared cube in which the point cloud to be included
       save_segy: bool - whether to save segy or .nc xarray
       segment_res_path - path to property for writing stored as (npy array)
       iline_path,, xline_path,, twt_path - pathes to dims for the cube conversion 
                                            stored as (npy array)
        npy arrays: iline_path,, xline_path, twt_path, segment_res_path must have the same length       
       """
    def __init__(self, save_fold, file_name, bolvanka_path, 
                segment_res_path, iline_path, 
                 xline_path, twt_path) -> None:
        self.save_fold = save_fold
        self.file_name = file_name
        # self.if_save_segy = save_segy
        self.segment_res = np.load(segment_res_path)
        self.iline_hor = np.load(iline_path)
        self.xline_hor = np.load(xline_path)
        self.twt_hor = np.load(twt_path)
        self.results_xr = xr.load_dataset(bolvanka_path)
        
        self.insert_res_to_xr()
        
    
    def insert_res_to_xr(self):
        print('Start inserting results to bolvanka xarray!')
        var_name = list(self.results_xr.data_vars.keys())[0]
        
        # Get the data array - make sure it's 3D
        # data_array = self.results_xr
        data_array = self.results_xr[var_name]
        
        print(f"Data array shape: {data_array.shape}")
        print(f"Data array dims: {data_array.dims}")
        
        # Check if data_array is actually 3D
        if len(data_array.shape) != 3:
            raise ValueError(f"Expected 3D array, got {len(data_array.shape)}D array with shape {data_array.shape}")
        
        # Get coordinate values
        iline_coords = self.results_xr['iline'].values
        xline_coords = self.results_xr['xline'].values
        twt_coords = self.results_xr['twt'].values
        
        # Verify coordinate dimensions
        print(f"iline coords shape: {iline_coords.shape}")
        print(f"xline coords shape: {xline_coords.shape}")
        print(f"twt coords shape: {twt_coords.shape}")
        
        # Use direct indexing - assume horizon coordinates are already indices
        # If iline_hor, xline_hor, twt_hor are actual coordinate values, convert to indices
        try:
            # Option 1: If they're already indices (0-based integer positions)
            data_array.values[self.iline_hor, self.xline_hor, self.twt_hor] = self.segment_res
            print(f"Option 1: Direct indexing successful")
            
        except (IndexError, TypeError):
            # Option 2: If they're coordinate values, need to convert to indices
            print("Converting coordinate values to indices...")
            
            # Create mapping dictionaries for fast lookup
            iline_dict = {val: i for i, val in enumerate(iline_coords)}
            xline_dict = {val: i for i, val in enumerate(xline_coords)}
            twt_dict = {val: i for i, val in enumerate(twt_coords)}
            
            # Convert horizon coordinates to indices
            iline_indices = np.array([iline_dict.get(val, -1) for val in self.iline_hor])
            xline_indices = np.array([xline_dict.get(val, -1) for val in self.xline_hor])
            twt_indices = np.array([twt_dict.get(val, -1) for val in self.twt_hor])
            
            # Filter out invalid indices
            valid_mask = (iline_indices >= 0) & (xline_indices >= 0) & (twt_indices >= 0)
            
            if not valid_mask.all():
                print(f"Warning: {len(valid_mask) - valid_mask.sum()} points have invalid coordinates")
            
            # Insert only valid points
            data_array.values[
                iline_indices[valid_mask], 
                xline_indices[valid_mask], 
                twt_indices[valid_mask]
            ] = self.segment_res[valid_mask]
            
            print(f"Option 2: Inserted {valid_mask.sum()} values")
        
        print('Finished inserting results!')
        
    
    def save_result(self, cube_format='segy'):
        if cube_format != 'segy':
            self.save_xarr()
        else:
            self.save_segy()

    def save_segy(self):
        print('Save SEGY')
        self.results_xr.attrs['coord_scalar'] = 1.00000
        self.results_xr.attrs['sample_rate'] = 2.0000
        self.results_xr.attrs['source_file'] = 'Cube_resulting'
        cube_path = os.path.join(self.save_fold, f'{self.file_name}.segy')
        # the dictionary with byte location of the coords
        trace_header_map = dict(iline=189,
                         xline=193,
                         cdp_x=73,
                         cdp_y=77)
        segy_writer(
            self.results_xr,
            cube_path,
            trace_header_map=trace_header_map
        )

    def save_xarr(self):
        print('Save xarray')
        # seism_cube.to_netcdf(os.path.join(xarray_fold, f'{cube.split(" ")[0]}.nc'), engine='h5netcdf')
        cube_path = os.path.join(self.save_fold, f'{self.file_name}.nc')
        self.results_xr.to_netcdf(cube_path, engine='h5netcdf')

def create_template_cube(cube_path, template_path):
    """
    Create a template (bolvanka) cube with all values set to -1.
    
    Parameters:
    -----------
    cube_path : str
        Path to original seismic cube (.nc file)
    template_path : str
        Path to save template cube
    """
    
    # Original cube
    cube_ds = xr.open_dataset(cube_path)
    
    # Get data variable name
    data_var_name = list(cube_ds.data_vars)[0]
    
    # Create a new DataArray with same shape/dims/coords but values = -1
    template_data = xr.DataArray(
        data=np.full_like(cube_ds[data_var_name].values, -1, dtype=np.int32),
        dims=cube_ds[data_var_name].dims,
        coords=cube_ds[data_var_name].coords,
        attrs=cube_ds[data_var_name].attrs
    )
    
    # Create new Dataset
    template_ds = xr.Dataset({data_var_name: template_data})
    
    # Copy all coordinates
    for coord in cube_ds.coords:
        if coord not in template_ds.coords:
            template_ds[coord] = cube_ds[coord]
    
    # Copy global attributes
    template_ds.attrs = cube_ds.attrs
    
    # Save
    template_ds.to_netcdf(template_path, engine='h5netcdf')
    print(f"Template cube saved to: {template_path}")
    print(f"Shape: {template_data.shape}")
    print(f"All values set to: {template_data.values[0,0,0]}")
    
    cube_ds.close()
    return template_ds