### JUST HELPFUL LOOP HOW TO CROP ALL CUBES USING SURFACES
### ALL CUBES HAVE DIFFERENT NUMBER OF TRACES AND THEY HAVE
### DIFFERENT XY VALS (SMALL DIFF < DX&DY) IN THE SAME TRACES.
### THE CODE MAKE ALL COORDS THE SAME AND SELECT ONLY THOSE TRACES
### THAT AARE PRESENT IN ALL CUBES

# import take_horizon_by_cdp

## Take one cube as reference one to transfer cdp_x, cdp_y values 
# comm_ilines = np.load(os.path.join(r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays', 'common_ilines.npy'))
# comm_xlines = np.load(os.path.join(r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays', 'common_xlines.npy'))
# surf_path = r'C:\Damir\unsupervised_seism_segment_novatek\unziped'
# upper_surf = pd.read_csv(os.path.join(surf_path, 'BU15 (TWT)txt'), delimiter=' ', header=None)
# lower_surf = pd.read_csv(os.path.join(surf_path, 'B_Time.grd (TWT)txt'), delimiter=' ', header=None)
# refer_cube_name = 'Cube_offset_ENV950.nc'
# refer_cube = xr.load_dataset(os.path.join(r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays',
#                              refer_cube_name))
# refer_cube = refer_cube.sel({'iline': comm_ilines, 'xline': comm_xlines})

## The surfaces must have the columns:
# upper_surf.columns = ['cdp_x', 'cdp_y', 'twt']
# lower_surf.columns = ['cdp_x', 'cdp_y', 'twt']

# seism_folder = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays'
# seism_folder_save = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes'
# cube_names = os.listdir(seism_folder)
# # xarray_fold = './cubes_as_xarrays'
# dct_stats = {}
# common_xlines = {}
# common_inines = {}
# for cube in cube_names:
#     gc.collect()
#     if ('.nc' not in cube) or (refer_cube_name in cube):
#         continue
#     f_path_cube = os.path.join(seism_folder, cube)
#     seism_cube = xr.load_dataset(f_path_cube)
#     # cut on intersection traces
#     seism_cube = seism_cube.sel({'iline': comm_ilines, 'xline': comm_xlines})
#     # assign the same coord vakue (there is a little mismatch < xy_step)
#     seism_cube['cdp_x'] = (('iline', 'xline'), refer_cube.cdp_x.values)
#     seism_cube['cdp_y'] = (('iline', 'xline'), refer_cube.cdp_y.values)
#     # take only horizones
#     seism_cube = take_horizon_by_cdp(seism_cube, upper_surf, 
#                                      lower_surf, padding_twt=2, 
#                                      padding_xy=26)
#     # save the cube
#     seism_cube.to_netcdf(os.path.join(seism_folder_save, 
#                                       f'{cube.split(".")[0]}_horzn.nc'), 
#                                       engine='h5netcdf')
#     print(f'cube {cube} saved successfulle!')

## save reference cube as well
# refer_cube = take_horizon_by_cdp(refer_cube, upper_surf, 
#                                  lower_surf, padding_twt=2, 
#                                  padding_xy=26)
# refer_cube.to_netcdf(os.path.join(seism_folder_save, 
#                                       f'{refer_cube_name.split(".")[0]}_horzn.nc'), 
#                                       engine='h5netcdf')