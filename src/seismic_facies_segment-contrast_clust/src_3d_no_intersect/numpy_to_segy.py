import numpy as np
import os
import xarray as xr
from segysak.segy import segy_writer
import zipfile

def save_segy(results_xr, save_path):
    print('Save SEGY')
    results_xr.attrs['coord_scalar'] = 1.00000
    results_xr.attrs['sample_rate'] = 2.0000
    results_xr.attrs['source_file'] = 'Cube_resulting'
    # cube_path = os.path.join(save_fold, f'{f_name}.segy')
    # the dictionary with byte location of the coords
    
    trace_header_map = dict(iline=189,
                         xline=193,
                         cdp_x=73,
                         cdp_y=77)
    segy_writer(
            results_xr,
            save_path,
            trace_header_map=trace_header_map
        )
    
def save_segy_from_np_cube(dataset_path, numpy_path, save_path):
    
    # Load dataset and numpy array
    ds = xr.open_dataset(dataset_path)
    data_array = np.load(numpy_path)
    
    print(f"Replacing existing 'data' variable with new values")
    ds['data'] = (ds.dims, data_array)  
    print(ds.dims)
    # Generate output filename and path
    # f_name = os.path.split(numpy_path)[1].split(".")[0]
    # fold_path = os.path.split(numpy_path)[0]
    
    # Save as SEGY
    save_segy(ds, save_path)
    

cube_path = r'../../data/full_size_npy/big_facies_categoric.npy'
bolvanka_path = r'../../data/full_size_cubes/bolvanka_head_init_resol.nc'
save_path = r'../../data/full_size_npy/big_facies_categoric.segy'

save_segy_from_np_cube(bolvanka_path, cube_path, save_path)
fold_path, filename = os.path.split(save_path)
zip_filename  = os.path.join(fold_path, f"{filename.split('.')[0]}.zip")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(save_path, arcname=os.path.basename(save_path))

os.remove(save_path)
# cube = np.load(cube_path)
# print(f'Nan number: {np.sum(np.isnan(cube))}')
# cube[np.isnan(cube)] = -1



# assert {np.sum(np.isnan(cube))}, 'there are nans anyway'

# np.save(cube_path, cube)