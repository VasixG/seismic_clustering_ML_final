import gc
from src.save_p_cloud_as_cube import SavePredictionPCloud
import pandas as pd
import numpy as np
import os
# from src.save_p_cloud_as_cube import SavePredictionPCloud
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import sklearn.cluster as clust
from sklearn.mixture import GaussianMixture
# from time import time
from two_stg_clust import TwoStageClust
from visualization import save_sec_interest

# def save_section(result_xarr, save_fold, name, sec=2550):
#     fig, ax = plt.subplots(1)
#     im1 = ax.scatter(result_xarr.sel({'twt': sec}).cdp_x.data,
#             result_xarr.sel({'twt': sec}).cdp_y.data,
#             c=result_xarr.sel({'twt': sec}).data, 
#             s=mpl.rcParams['lines.markersize']/4,
#             rasterized=True)
#     cbar = fig.colorbar(im1)
#     fig.savefig(os.path.join(save_fold, f'{name}.pdf'))
#     plt.close()

data_folder = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\p_clouds_dwn2'

def check_weight_names(weights:dict, 
                       data_folder):
    for w in weights.keys():
        assert w in os.listdir(data_folder), f'w_name: {w} is not in data folder'

weights = {'cdp_x.npy': 0.25,
           'cdp_y.npy': 0.25,
           'twt.npy': 0.5}

check_weight_names(weights, data_folder)

use_spatial = ['all', 'only_twt', 'none']

twt_weights = [0.5, 1]

# attrib_config = ['only_spectr', 'no_spectr', 
#                  'all', 'only_geom']
attrib_config = ['only_geom']

clust_methods = {
                'gmm3': 3,
                 'gmm4': 4,
                 'gmm5': 5,
                 
                 }
# n_compons = [5, 10, 20, 40]

# save_f = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\cluster_res\gmm_4_less_spat_w'
bolvanka_path = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\bolvanka.nc'
# segment_res_pth = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\cluster_res\gmm_4_clust_less_spat_w.npy'
iline_path = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\p_clouds_dwn2\iline.npy'
xline_path = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\p_clouds_dwn2\xline.npy'
twt_path = r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\p_clouds_dwn2\twt.npy'
res_fold =r'C:\Damir\unsupervised_seism_segment_novatek\cubes_as_xarrays\horizon_cubes\cluster_res\geom_attrs_experim'

# for us_spat in use_spatial:
    # for use_only_time:
        
# for attr_conf in attrib_config: 

for alg_name, n_comp in clust_methods.items():
    for us_spat in use_spatial:
        if us_spat != 'none':
            for twt_w in twt_weights:
                try:
                    # name = (f'{alg_name}_{attr_conf}_use_spat' if use_spatial[0]
                            # else f'{alg_name}_{attr_conf}_no_spat')
                    name = f'{alg_name}_{attrib_config[0]}_twt_w_{twt_w}'
                    if us_spat == 'all':
                        name+= '_0.25xy'
                    print(f'Start fit-predict for {name} alg')
                    
                    weights['twt.npy'] = twt_w
                    two_stg_clust = TwoStageClust(data_folder=data_folder,
                                                  )
                    two_stg_clust.load_features(use_spatial=us_spat,
                                                attrib_config=attrib_config[0],
                                                weights=weights)
                    clust_alg = GaussianMixture(n_comp)
                    res = two_stg_clust.fit_predict(clust_alg,
                                                    use_kmeans_centr=False)
                    print(f'End fit-predict for {name} alg')

                    save_f = os.path.join(res_fold, name)
                    os.mkdir(save_f)
                    segment_res_path = os.path.join(save_f, f'{name}.npy')
                    np.save(segment_res_path, res)
                    saver = SavePredictionPCloud(save_f, name, bolvanka_path,
                                                segment_res_path=segment_res_path, 
                                                iline_path=iline_path, 
                                                xline_path=xline_path,
                                                twt_path=twt_path)
                    saver.save_segy()
                    # saver.save_xarr()
                    # save_section(saver.results_xr, save_f, name, sec=2568)
                    save_sec_interest(saver.results_xr, save_f)
                except (ValueError, RuntimeError, MemoryError, TypeError, AttributeError) as e:
                    print(f'{name} has error: {e} continue')
                    continue
                finally:
                    gc.collect()
        else:
            try:
                    # name = (f'{alg_name}_{attr_conf}_use_spat' if use_spatial[0]
                            # else f'{alg_name}_{attr_conf}_no_spat')
                    name = f'{alg_name}_{attrib_config[0]}_no_spat'
                    
                    print(f'Start fit-predict for {name} alg')
                    two_stg_clust = TwoStageClust(data_folder=data_folder,
                                                  )
                    two_stg_clust.load_features(use_spatial=us_spat,
                                                attrib_config=attrib_config[0],
                                                weights=None)
                    clust_alg = GaussianMixture(n_comp)
                    res = two_stg_clust.fit_predict(clust_alg,
                                                    use_kmeans_centr=False)
                    print(f'End fit-predict for {name} alg')

                    save_f = os.path.join(res_fold, name)
                    os.mkdir(save_f)
                    segment_res_path = os.path.join(save_f, f'{name}.npy')
                    np.save(segment_res_path, res)
                    saver = SavePredictionPCloud(save_f, name, bolvanka_path,
                                                segment_res_path=segment_res_path, 
                                                iline_path=iline_path, 
                                                xline_path=xline_path,
                                                twt_path=twt_path)
                    saver.save_segy()
                    # saver.save_xarr()
                    # save_section(saver.results_xr, save_f, name, sec=2568)
                    save_sec_interest(saver.results_xr, save_f)
            except (ValueError, RuntimeError, MemoryError, TypeError, AttributeError) as e:
                print(f'{name} has error: {e} continue')
                continue
            finally:
                gc.collect()