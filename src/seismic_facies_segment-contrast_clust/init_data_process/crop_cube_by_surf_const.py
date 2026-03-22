
import os
import pandas as pd
from segysak.segy import segy_loader
from segysak.segy import segy_writer
import xarray as xr
import numpy as np
import py7zr

def take_horizon_by_cdp(cube, upper_surf, lower_surf, 
                        padding_twt=2, padding_xy=26):
    """
    Cropp with a constant init cube using two surfaces at xy and twt dims
    the result is also cube but restricted with surfaces coords + padding 
    """
    
    # Get surface bounds in cdp
    cdp_x_min = min(upper_surf.cdp_x.min(), lower_surf.cdp_x.min()) - padding_xy
    cdp_x_max = max(upper_surf.cdp_x.max(), lower_surf.cdp_x.max()) + padding_xy
    cdp_y_min = min(upper_surf.cdp_y.min(), lower_surf.cdp_y.min()) - padding_xy
    cdp_y_max = max(upper_surf.cdp_y.max(), lower_surf.cdp_y.max()) + padding_xy
    
    # Find which iline/xline indices correspond to these cdp bounds
    cdp_x_mask = (cube.cdp_x >= cdp_x_min) & (cube.cdp_x <= cdp_x_max)
    cdp_y_mask = (cube.cdp_y >= cdp_y_min) & (cube.cdp_y <= cdp_y_max)
    spatial_mask = cdp_x_mask & cdp_y_mask
    
    # Get iline/xline indices from the mask
    iline_has_data = spatial_mask.any(dim='xline')
    xline_has_data = spatial_mask.any(dim='iline')
    
    # Get the actual iline/xline
    iline_selection = cube.iline.where(iline_has_data, drop=True).values
    xline_selection = cube.xline.where(xline_has_data, drop=True).values
    
    # Apply additional padding in iline/xline space
    iline_min = iline_selection.min() - padding_xy
    iline_max = iline_selection.max() + padding_xy
    xline_min = xline_selection.min() - padding_xy
    xline_max = xline_selection.max() + padding_xy
    
    # Time bounds from surfaces
    twt_min = min(upper_surf.twt.min(), lower_surf.twt.min()) - padding_twt
    twt_max = max(upper_surf.twt.max(), lower_surf.twt.max()) + padding_twt
    
    iline_min = max(iline_min, cube.iline.values.min())
    iline_max = min(iline_max, cube.iline.values.max())
    xline_min = max(xline_min, cube.xline.values.min())
    xline_max = min(xline_max, cube.xline.values.max())
    twt_min = max(twt_min, cube.twt.values.min())
    twt_max = min(twt_max, cube.twt.values.max())
    
    # Extract subcube
    subcube = cube.sel(
        iline=slice(iline_min, iline_max),
        xline=slice(xline_min, xline_max),
        twt=slice(twt_min, twt_max)
    )
    
    print(f"Ilines [{iline_min}:{iline_max}], "
          f"xlines [{xline_min}:{xline_max}], "
          f"twt [{twt_min}:{twt_max}]")
    
    return subcube


def write_segy(cube_xarr, path):
    """convert xarray cube into .segy and save it"""
    
    # necessary attributes
    cube_xarr.attrs['coord_scalar'] = 1.00000
    cube_xarr.attrs['sample_rate'] = 2.0000
    cube_xarr.attrs['source_file'] = 'Cube_resulting'

    segy_writer(
                cube_xarr,
                path,
                # trace_header_map=dict(iline=5, xline=21)
            )
    
    return None

# trace1.data.plot(x='xline', y='twt', yincrease=False)

def save_xarray(cube_xarr, path):
    """just a small function how to save xarray with my environment
       properly for the further reading"""
    cube_xarr.to_netcdf(path, engine='h5netcdf')


# def extract_7z(archive_path, output_directory, password=None):
#     """
#     Function for extracting a single .7z array
#     """
#     try:
#         with py7zr.SevenZipFile(archive_path, mode='r', password=password) as z:
#             z.extractall(path=output_directory)
#         print(f"Successfully extracted '{archive_path}' to '{output_directory}'")
#     except py7zr.Bad7zFile:
#         print(f"Error: '{archive_path}' is not a valid 7z file or is corrupted.")
#     except py7zr.PasswordRequired:
#         print(f"Error: Password required for '{archive_path}'. Please provide the correct password.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
