import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree
import warnings
import time
from numba import jit, prange
import gc
import os

warnings.filterwarnings('ignore')

def extract_seismic_between_surfaces_fixed(cube_dataarray, upper_surface_df, lower_surface_df,
                                          search_distance=50.0,  # Use full surface spacing
                                          remove_zero_traces=True):
    """
    Function for taking observations from cube which lay between seismic surfaces
    and save it as point cloud
    """
    
    print("=" * 80)
    print("FIXED: EXTRACTING SEISMIC BETWEEN SURFACES")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Prepare surfaces
    print("\n1. Preparing surfaces...")
    
    # Validate surfaces have same XY
    assert np.array_equal(upper_surface_df[['cdp_x', 'cdp_y']].values,
                         lower_surface_df[['cdp_x', 'cdp_y']].values), \
        "Surfaces must have identical XY coordinates!"
    
    # Create merged surface
    surface_df = pd.DataFrame({
        'cdp_x': upper_surface_df['cdp_x'].values,
        'cdp_y': upper_surface_df['cdp_y'].values,
        'twt_upper': upper_surface_df['twt'].values,
        'twt_lower': lower_surface_df['twt'].values
    })
    
    # Filter valid surfaces (positive thickness)
    surface_df = surface_df[surface_df['twt_lower'] > surface_df['twt_upper']].copy()
    
    if len(surface_df) == 0:
        raise ValueError("No valid surface points!")
    
    print(f"   Valid surface points: {len(surface_df):,}")
    print(f"   Surface spacing: ~{abs(surface_df['cdp_x'].iloc[1] - surface_df['cdp_x'].iloc[0]):.1f} m")
    
    # Step 2: Get ALL cube trace coordinates
    print("\n2. Getting ALL cube trace coordinates...")
    
    # Get actual iline/xline numbers
    iline_coords = cube_dataarray.iline.values.astype(np.int32)
    xline_coords = cube_dataarray.xline.values.astype(np.int32)
    
    # Create 2D arrays of all trace positions
    n_ilines, n_xlines = len(iline_coords), len(xline_coords)
    iline_indices, xline_indices = np.meshgrid(np.arange(n_ilines), 
                                              np.arange(n_xlines), 
                                              indexing='ij')
    iline_indices = iline_indices.ravel()  # 0-based indices for array access
    xline_indices = xline_indices.ravel()
    
    # Get CDP coordinates for ALL traces
    if cube_dataarray.cdp_x.ndim == 1:
        cdp_x_1d = cube_dataarray.cdp_x.values
        cdp_y_1d = cube_dataarray.cdp_y.values
        cdp_x_grid, cdp_y_grid = np.meshgrid(cdp_x_1d, cdp_y_1d, indexing='ij')
    else:
        cdp_x_grid = cube_dataarray.cdp_x.values
        cdp_y_grid = cube_dataarray.cdp_y.values
    
    all_trace_coords = np.column_stack([
        cdp_x_grid[iline_indices, xline_indices],
        cdp_y_grid[iline_indices, xline_indices]
    ])
    
    print(f"   Total cube traces: {len(all_trace_coords):,}")
    print(f"   Cube trace spacing: ~{abs(cdp_x_grid[0,1] - cdp_x_grid[0,0]):.1f} m")
    
    # Step 3: Build KDTree for ALL surface points
    print("\n3. Building spatial index...")
    
    surface_points = surface_df[['cdp_x', 'cdp_y']].values
    surface_tree = KDTree(surface_points)
    
    # Step 4: Find nearest surface point for EVERY cube trace
    print(f"\n4. Finding nearest surface for ALL traces (search radius: {search_distance:.1f} m)...")
    
    # Query for nearest surface point
    distances, surface_indices = surface_tree.query(all_trace_coords, k=1)
    
    print(f"   Minimum distance: {distances.min():.2f} m")
    print(f"   Maximum distance: {distances.max():.2f} m")
    print(f"   Median distance: {np.median(distances):.2f} m")
    
    # Step 5: Get surface TWT values for each trace
    print("\n5. Getting surface TWT values...")
    
    upper_twt_all = surface_df.iloc[surface_indices]['twt_upper'].values
    lower_twt_all = surface_df.iloc[surface_indices]['twt_lower'].values
    
    # Step 6: Remove traces with invalid surface pairs
    print("\n6. Filtering traces with valid surfaces...")
    
    valid_mask = (upper_twt_all < lower_twt_all)  # Upper must be above lower
    valid_mask &= (~np.isnan(upper_twt_all)) & (~np.isnan(lower_twt_all))
    
    valid_iline_idx = iline_indices[valid_mask]
    valid_xline_idx = xline_indices[valid_mask]
    valid_upper_twt = upper_twt_all[valid_mask]
    valid_lower_twt = lower_twt_all[valid_mask]
    valid_distances = distances[valid_mask]
    
    print(f"   Traces with valid surfaces: {len(valid_iline_idx):,} "
          f"({len(valid_iline_idx)/len(iline_indices)*100:.1f}%)")
    
    if len(valid_iline_idx) == 0:
        print("   ERROR: No traces found with valid surface pairs!")
        print("   Check if surfaces cover the cube area.")
        return pd.DataFrame()
    
    # Step 7: Get cube data and TWT coordinates
    print("\n7. Loading cube data...")
    
    data_3d = cube_dataarray.values.astype(np.float32)
    twt_coords = cube_dataarray.twt.values.astype(np.float32)
    
    # Step 8: Extract seismic samples
    print("\n8. Extracting seismic samples...")
    
    all_results = []
    
    # Process in batches to avoid memory issues
    batch_size = 100000
    total_extracted = 0
    
    for i in range(0, len(valid_iline_idx), batch_size):
        batch_end = min(i + batch_size, len(valid_iline_idx))
        
        batch_iline_idx = valid_iline_idx[i:batch_end]
        batch_xline_idx = valid_xline_idx[i:batch_end]
        batch_upper_twt = valid_upper_twt[i:batch_end]
        batch_lower_twt = valid_lower_twt[i:batch_end]
        batch_distances = valid_distances[i:batch_end]
        
        batch_points = []
        
        for j in range(len(batch_iline_idx)):
            iline_idx = batch_iline_idx[j]
            xline_idx = batch_xline_idx[j]
            upper_twt = batch_upper_twt[j]
            lower_twt = batch_lower_twt[j]
            distance = batch_distances[j]
            
            # Find TWT indices
            upper_idx = np.searchsorted(twt_coords, upper_twt, side='left')
            lower_idx = np.searchsorted(twt_coords, lower_twt, side='right')
            
            upper_idx = max(0, upper_idx)
            lower_idx = min(len(twt_coords), lower_idx)
            
            if upper_idx < lower_idx:
                # Extract seismic data
                trace_data = data_3d[iline_idx, xline_idx, upper_idx:lower_idx]
                
                # Skip if all zeros (but keep if some non-zero values)
                if np.all(np.abs(trace_data) < 1e-10):
                    continue
                
                trace_twt = twt_coords[upper_idx:lower_idx]
                n_points = len(trace_data)
                
                # Create DataFrame for this trace
                trace_df = pd.DataFrame({
                    'cdp_x': np.full(n_points, cdp_x_grid[iline_idx, xline_idx]),
                    'cdp_y': np.full(n_points, cdp_y_grid[iline_idx, xline_idx]),
                    'iline': np.full(n_points, iline_coords[iline_idx]),
                    'xline': np.full(n_points, xline_coords[xline_idx]),
                    'twt': trace_twt,
                    'data': trace_data,
                    'distance_to_surface': np.full(n_points, distance)
                })
                
                batch_points.append(trace_df)
        
        if batch_points:
            batch_df = pd.concat(batch_points, ignore_index=True)
            all_results.append(batch_df)
            total_extracted += len(batch_df)
        
        print(f"   Processed {batch_end}/{len(valid_iline_idx)} traces, "
              f"extracted {total_extracted:,} points")
    
    # Combine all results
    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
    else:
        result_df = pd.DataFrame()
    
    # Step 9: Final statistics
    print("\n9. Final results...")
    
    if len(result_df) == 0:
        print("   ERROR: No data extracted!")
        print("   Possible issues:")
        print("   1. Surfaces don't overlap with cube area")
        print("   2. Surface TWT values outside cube TWT range")
        print("   3. All extracted traces have zero values")
        return result_df
    
    # Calculate coverage
    unique_traces = result_df[['iline', 'xline']].drop_duplicates()
    coverage_pct = len(unique_traces) / len(valid_iline_idx) * 100
    
    print(f"   Total extracted points: {len(result_df):,}")
    print(f"   Unique traces with data: {len(unique_traces):,}")
    print(f"   Coverage of valid traces: {coverage_pct:.1f}%")
    print(f"   Data range: [{result_df['data'].min():.6f}, {result_df['data'].max():.6f}]")
    print(f"   TWT range: [{result_df['twt'].min():.1f}, {result_df['twt'].max():.1f}] ms")
    
    # Check for zeros
    non_zero_mask = np.abs(result_df['data'].values) > 1e-10
    zero_pct = (1 - non_zero_mask.sum() / len(result_df)) * 100
    
    if zero_pct > 50:
        print(f"   WARNING: {zero_pct:.1f}% of extracted values are near-zero!")
        print(f"   You may want to filter these: result_df = result_df[np.abs(result_df['data']) > 1e-10]")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED IN {elapsed_time:.2f} SECONDS")
    print(f"{'='*80}")
    
    return result_df

def check_data_coverage(cube_dataarray, upper_surface_df, lower_surface_df):
    """
    Diagnostic function to check why extraction might fail.
    """
    print("\n" + "="*80)
    print("DATA COVERAGE DIAGNOSTICS")
    print("="*80)
    
    # Get cube bounds
    if cube_dataarray.cdp_x.ndim == 1:
        cdp_x_min, cdp_x_max = cube_dataarray.cdp_x.values.min(), cube_dataarray.cdp_x.values.max()
        cdp_y_min, cdp_y_max = cube_dataarray.cdp_y.values.min(), cube_dataarray.cdp_y.values.max()
    else:
        cdp_x_min, cdp_x_max = cube_dataarray.cdp_x.values.min(), cube_dataarray.cdp_x.values.max()
        cdp_y_min, cdp_y_max = cube_dataarray.cdp_y.values.min(), cube_dataarray.cdp_y.values.max()
    
    # Get surface bounds
    surf_x_min, surf_x_max = upper_surface_df['cdp_x'].min(), upper_surface_df['cdp_x'].max()
    surf_y_min, surf_y_max = upper_surface_df['cdp_y'].min(), upper_surface_df['cdp_y'].max()
    
    # Get TWT ranges
    cube_twt_min, cube_twt_max = cube_dataarray.twt.values.min(), cube_dataarray.twt.values.max()
    upper_twt_min, upper_twt_max = upper_surface_df['twt'].min(), upper_surface_df['twt'].max()
    lower_twt_min, lower_twt_max = lower_surface_df['twt'].min(), lower_surface_df['twt'].max()
    
    print(f"\nSPATIAL COVERAGE:")
    print(f"  Cube X range: {cdp_x_min:.0f} to {cdp_x_max:.0f}")
    print(f"  Surface X range: {surf_x_min:.0f} to {surf_x_max:.0f}")
    print(f"  Cube Y range: {cdp_y_min:.0f} to {cdp_y_max:.0f}")
    print(f"  Surface Y range: {surf_y_min:.0f} to {surf_y_max:.0f}")
    
    print(f"\nTWT COVERAGE:")
    print(f"  Cube TWT: {cube_twt_min:.1f} to {cube_twt_max:.1f} ms")
    print(f"  Upper surface: {upper_twt_min:.1f} to {upper_twt_max:.1f} ms")
    print(f"  Lower surface: {lower_twt_min:.1f} to {lower_twt_max:.1f} ms")
    
    # Check overlap
    x_overlap = max(0, min(cdp_x_max, surf_x_max) - max(cdp_x_min, surf_x_min))
    y_overlap = max(0, min(cdp_y_max, surf_y_max) - max(cdp_y_min, surf_y_min))
    
    print(f"\nOVERLAP ANALYSIS:")
    print(f"  X overlap: {x_overlap:.0f} m ({x_overlap/(cdp_x_max-cdp_x_min)*100:.1f}% of cube)")
    print(f"  Y overlap: {y_overlap:.0f} m ({y_overlap/(cdp_y_max-cdp_y_min)*100:.1f}% of cube)")
    
    # Check TWT containment
    upper_in_cube = (upper_twt_min >= cube_twt_min) and (upper_twt_max <= cube_twt_max)
    lower_in_cube = (lower_twt_min >= cube_twt_min) and (lower_twt_max <= cube_twt_max)
    
    print(f"\nTWT CONTAINMENT:")
    print(f"  Upper surface in cube TWT range: {upper_in_cube}")
    print(f"  Lower surface in cube TWT range: {lower_in_cube}")
    
    if not upper_in_cube or not lower_in_cube:
        print(f"  WARNING: Surfaces may be outside cube TWT range!")
        print(f"  Consider checking surface TWT values.")
    
    # Quick sample of surface points
    print(f"\nSURFACE SAMPLE (first 5 points):")
    print("  CDP_X, CDP_Y, Upper_TWT, Lower_TWT:")
    for i in range(min(5, len(upper_surface_df))):
        print(f"  {upper_surface_df['cdp_x'].iloc[i]:.0f}, "
              f"{upper_surface_df['cdp_y'].iloc[i]:.0f}, "
              f"{upper_surface_df['twt'].iloc[i]:.1f}, "
              f"{lower_surface_df['twt'].iloc[i]:.1f}")

# Main function with diagnostics
def extract_with_diagnostics(cube_path, upper_surface_path, lower_surface_path,
                             search_distance=50.0, remove_zero_traces=True):
    """
    Complete extraction with diagnostics.
    """
    print(f"Loading cube: {cube_path}")
    cube_ds = xr.open_dataset(cube_path)
    cube_data = cube_ds[list(cube_ds.data_vars)[0]]
    
    print(f"Loading surfaces...")
    upper_surface = pd.read_csv(upper_surface_path)
    lower_surface = pd.read_csv(lower_surface_path)
    
    # Run diagnostics first
    check_data_coverage(cube_data, upper_surface, lower_surface)
    
    # Extract data
    result_df = extract_seismic_between_surfaces_fixed(
        cube_dataarray=cube_data,
        upper_surface_df=upper_surface,
        lower_surface_df=lower_surface,
        search_distance=search_distance,  # Full surface spacing
        remove_zero_traces=remove_zero_traces
    )
    
    
    return result_df

# Simple test function
def quick_test_extraction(cube_dataarray, upper_surface_df, lower_surface_df):
    """
    Quick test on a small subset.
    """
    print("\n" + "="*80)
    print("QUICK TEST ON SUBSET")
    print("="*80)
    
    # Take first 1000 surface points
    test_upper = upper_surface_df.head(1000).copy()
    test_lower = lower_surface_df.head(1000).copy()
    
    result = extract_seismic_between_surfaces_fixed(
        cube_dataarray=cube_dataarray,
        upper_surface_df=test_upper,
        lower_surface_df=test_lower,
        search_distance=50.0,
        remove_zero_traces=True
    )
    
    if len(result) > 0:
        print(f"\nTest successful! Extracted {len(result):,} points.")
        print(f"Data range: {result['data'].min():.6f} to {result['data'].max():.6f}")
        print(f"Unique traces: {result[['iline', 'xline']].drop_duplicates().shape[0]}")
    else:
        print("\nTest failed - no data extracted.")
    
    return result


# surf_path = r'C:\Damir\unsupervised_seism_segment_novatek\unziped'
# cube_path = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\cubes_xarrays\Cube_2019_15Hz_horzn.nc'
# upper_surface_path = os.path.join(surf_path, 'upper_surf.csv')
# lower_surface_path = os.path.join(surf_path, 'lower_surf.csv')

# # cube_ds = xr.open_dataset(cube_path)
# # cube_data = cube_ds[list(cube_ds.data_vars)[0]]
# # upper = pd.read_csv(upper_surface_path)
# # lower = pd.read_csv(lower_surface_path)
# test_result = extract_with_diagnostics(cube_path, upper_surface_path, lower_surface_path)

#VERY IMPORTANT FUNCTION!

def align_dataframes_by_intersection(dataframes):
    """
    Extract_with_diagnostics function takes attribute point clouds from cubes in df.
    The attributes have missing values, skipped seismic traces, difference in dx,dy
    due to the calculation. The function align it io the same shape for convinience

    Parameters:
    -----------
    dataframes : list of pd.DataFrame
        List of dataframes with columns: cdp_x, cdp_y, iline, xline, twt
        
    Returns:
    --------
    list of pd.DataFrame
        List of aligned dataframes with same observations
    """
    pos_cols = ['cdp_x', 'cdp_y', 'iline', 'xline', 'twt']
    print(f"Processing {len(dataframes)} dataframes...")
    out_folder = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\p_clouds_dwn2'
    # Step 1: Create sets of unique keys for each dataframe
    key_sets = []
    for i, df in enumerate(dataframes):
        gc.collect()
        keys = set(zip(df['iline'], df['xline']))
        key_sets.append(keys)
        print(f"Dataframe {i}: {len(df):,} rows, {len(keys):,} unique keys")

    # Step 2: Find intersection of all key sets
    intersection_keys = set.intersection(*key_sets)
    print(f"\nIntersection has {len(intersection_keys):,} unique keys")
    print(f"This is {len(intersection_keys)/len(key_sets[0])*100:.1f}% of smallest dataset")
    del key_sets
    if len(intersection_keys) == 0:
        raise ValueError("No common observations found across all dataframes!")
    
    # Step 3: Filter each dataframe to keep only intersection keys
    aligned_dataframes = []
    
    for i, df in enumerate(dataframes):
        # Create tuple keys for current dataframe
        df_keys = list(zip(df['iline'], df['xline']))
        
        # Create mask for rows that exist in intersection
        mask = [key in intersection_keys for key in df_keys]
        
        # Filter dataframe
        filtered_df = df[mask].copy().reset_index(drop=True)
        if i == 0:
            for col in pos_cols:
                np.save(os.path.join(out_folder, f'{col}.npy'),
                        filtered_df[col].values)
        np.save(os.path.join(out_folder, f'{filtered_df.columns[-1]}.npy'),
                        filtered_df[filtered_df.columns[-1]].values)
        # aligned_dataframes.append(filtered_df)
        print(f"Dataframe {i}: filtered to {len(filtered_df):,} rows")
    
    # Verify all dataframes have same number of rows
    final_lengths = [len(df) for df in aligned_dataframes]
    if len(set(final_lengths)) == 1:
        print(f"\n✓ Success! All dataframes aligned to {final_lengths[0]:,} rows")
    else:
        print(f"\n⚠ Warning: Dataframes have different lengths: {final_lengths}")
    
    return aligned_dataframes

