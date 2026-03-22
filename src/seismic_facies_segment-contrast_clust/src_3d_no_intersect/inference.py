import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Optional
import pandas as pd
import os
from utils import down_cube_by_sel_ix_line, save_segy
import xarray as xr
from surface_extractor import SeismicMapExtractor
from sklearn.metrics import adjusted_rand_score
from visualization import save_pic_only_horizon
from utils import read_or_create_metric

class Inference:

    def __init__(self, patch_size,
                 data_cl_inst,
                 result_fold, 
                 config,
                 trained_model,
                 device, 
                 run_study, 
                 trial):
        # self.patch_size = patch_size
        self.data_cl_inst = data_cl_inst
        self.config = config
        self.result_fold = result_fold
        self.loader = data_cl_inst.create_seismic_dataloaders(dxy=patch_size[0],
                                                         dt=patch_size[-1],
                                                         training=False)
        self.full_dataset = data_cl_inst.full_dataset 
        self.presence_mask = data_cl_inst.presence_mask.numpy()  
        self.trained_model = trained_model
        self.run_study = run_study
        self.trial = trial
        self.device = device
        # self.cluster_num = config['cluster_num']
        self.clust_options = None
        self._get_clust_options()
        self.read_bolvankas()
        self.read_surf_to_plot()
        self.inference_bs = config['inference_bs']
        # self.trained_model.eval()
        # self.trained_model.to(self.device)
        
        self.volume_shape = self.presence_mask.shape  # (H, W, T)
        self.cluster_num = None
        self.latent_vectors = None
        self.patch_coords = None #indecis to be precise
        self.cluster_labels = None
        self.clustered_volume = None
    
    def _get_clust_options(self):
       
        assert isinstance(self.config['cluster_num'], list), 'Cluster specification is not list'
        if len(self.config['cluster_num']) > 0:
            self.clust_options = self.config['cluster_num']
        
        else:
            raise ValueError(f'Cluster specification is incorrect! {self.config["cluster_num"]}')

    def get_latent_vecs(self):
        
        print("Extracting latent vectors from all patches...")
        
        all_latent_vectors = []
        all_coords = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.loader):
                
                attrs, categor_attrs = batch_data
                attrs = attrs.to(self.device)
                categor_attrs = categor_attrs.to(self.device) if categor_attrs is not None else None

                mu, logvar = self.trained_model.encode(attrs, categor_attrs)
                z = self.trained_model.reparameterize(mu, logvar)
                all_latent_vectors.append(z.cpu().numpy())
                
        self.latent_vectors = np.vstack(all_latent_vectors)
        print(f'Latent vecs shape: {self.latent_vectors.shape}')
        self.patch_coords = np.array(self.full_dataset.valid_patches)
        
        print(f"Extracted {len(self.latent_vectors)} latent vectors of dimension {self.latent_vectors.shape[1]}")

    def latent_space_clust(self):
        
        print(f"Fitting GMM with {self.cluster_num} clusters...")
        self.gmm = GaussianMixture(
            n_components=self.cluster_num,
            covariance_type='full', 
            max_iter=200,
            n_init=5
        )
        
        if self.config['intersect']:
            # fit on downsampled quantity 
            self.gmm = self.gmm.fit(self.latent_vectors[::8])
            self.cluster_labels = self.gmm.predict(self.latent_vectors)
        else:
            self.cluster_labels = self.gmm.fit_predict(self.latent_vectors)
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))
        print(f"Cluster sizes: {cluster_sizes}")
        
        # for cluster_id, size in cluster_sizes.items():
            # self.run_study[f"clustering/cluster_size_{cluster_id}_{self.trial.number}"].append(size)
        
        return self.cluster_labels
    
    def tsne_visual(self):
        if self.latent_vectors is None or self.cluster_labels is None:
            raise ValueError("Run latent_space_clust() first")
        
        print("Computing t-SNE...")
        
        lat_vec_sparse = (self.latent_vectors[::4] if not self.config['intersect'] 
                          else self.latent_vectors[::16])

        clust_lab_sparse = (self.cluster_labels[::4] if not self.config['intersect'] 
                          else self.cluster_labels[::16])
        n_samples = len(lat_vec_sparse)
        perplexity = min(30, n_samples - 1) 
        
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity
            # max_iter=600
        )

        unique_labels = np.unique(clust_lab_sparse)
        self.cluster_num = len(unique_labels)
        
        cmap = plt.cm.get_cmap('tab10', self.cluster_num)
        
        latent_2d = tsne.fit_transform(lat_vec_sparse)
        
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        
        scatter1 = axes.scatter(
            latent_2d[:, 0], 
            latent_2d[:, 1], 
            c=clust_lab_sparse, 
            cmap=cmap,
            s=10,
            alpha=0.7, rasterized=True
        )
        axes.set_title(f't-SNE of Latent Space (Clusters)')
        axes.set_xlabel('t-SNE Component 1')
        axes.set_ylabel('t-SNE Component 2')
        axes.grid(axis='both', linestyle='--', alpha=0.7)
        cbar = fig.colorbar(scatter1, ax=axes, ticks=unique_labels)
        cbar.set_label(label='cluster_ID',size=12)
        
        fig.tight_layout()
        
        # self.run_study[f"imgs/lat_space_tsne_{self.trial.number}"].upload(fig)
        self.run_study.upload(f'lat_space_tsne_{self.trial.number}.pdf', fig)
        
        # os.makedirs(f"{self.result_fold}/tsne", exist_ok=True)
        # plt.savefig(f"{self.result_fold}/tsne/latent_tsne_trial_{self.trial.number}.png", 
        #            dpi=150, bbox_inches='tight')
        
        print("t-SNE visualization saved and logged to Neptune")
        # plt.close(fig)

    
    def create_clustered_volume(self):
    
        if self.cluster_labels is None:
            raise ValueError("Run latent_space_clust() first")
        
        print("Creating clustered volume...")
        
        self.clustered_volume = np.full(self.volume_shape, -1, dtype=np.int32)
        
        patch_h, patch_w, patch_t = self.full_dataset.stride
        #---------------------------------------------------
        # # cube filling, no stride
        # self.clustered_volume = self.clustered_volume.flatten() 
        # batch_num = self.patch_coords // self.inference_bs
        # start_ind = 0
        # end_ind = self.inference_bs
        # for b_ind in range(batch_num):
        #     # batch_coord = self.patch_coords[start_ind:end_ind]
        #     batch_labels = self.cluster_labels[start_ind:end_ind]
        #     self.clustered_volume[batch_coord] = batch_labels
        #     start_ind+=self.inference_bs
        #     end_ind+=self.inference_bs
        # #last data portion < batch size
        # self.clustered_volume[end_ind:] = self.cluster_labels[end_ind:]
                
        # for i, (h, w, t) in enumerate(self.patch_coords):
        #     self.clustered_volume[
        #         h:h + patch_h,
        #         w:w + patch_w,
        #         t:t + patch_t
        #     ] = self.cluster_labels[i]
        #---------------------------------------------------
        
        for i, (h, w, t) in enumerate(self.patch_coords):
            self.clustered_volume[
                h:h + patch_h,
                w:w + patch_w,
                t:t + patch_t
            ] = self.cluster_labels[i]
        
        self.clustered_volume[self.presence_mask == 0] = -1
        unique, counts = np.unique(self.clustered_volume[self.presence_mask==1], 
                                   return_counts=True)
        volume_cluster_sizes = dict(zip(unique, counts))
        print(f"Volume cluster distribution: {volume_cluster_sizes}")
        
        # return self.clustered_volume
    
    def save_clust_cube(self, filename: Optional[str] = None):
        """
        Save the clustered volume as a numpy array
        """
        if self.clustered_volume is None:
            self.create_clustered_volume()
        
        # self.get_section_picture()
        
        if filename is None:
            filename = f"clusters_{self.cluster_num}_trial_{self.trial.number}"
        
        # save_path = os.path.join(self.result_fold, 'cubes', filename)
        save_fold = os.path.join(self.result_fold, 'cubes')
        
        #TODO: make conditional inference
        self.clustered_volume = self.clustered_volume.astype('int16')
        if self.config['save_segy_cube']:

            down_res = down_cube_by_sel_ix_line(self.bolvanka_full, 
                                                self.bolvanka_down, 
                                                self.clustered_volume)
            save_segy(down_res, save_fold, filename)
            
            print(f"Clustered volume for clust num: {self.cluster_num} is saved")
        if self.config['save_np_cube']:
            np.save(os.path.join(save_fold, f'{filename}.npy'), 
                                  self.clustered_volume)
    
    def clust_cube_process(self, class_num):
        extractor = SeismicMapExtractor(max_distance=50.0, use_dask=True)
        
        ind_map_path = self.config['data_paths']['index_map']
        surf_points_path = self.config['data_paths']['surface_points']
        facies_mask = np.load(self.config['data_paths']['facies_mask'])
        section_conf = self.config['cross_section']
        
        extractor.load_index_map(ind_map_path)
        extractor.read_surface_points(surf_points_path)
        if self.clustered_volume.shape[0] < 1000: #TODO remove the hardcode
            xr_cube = self.bolvanka_down.copy()
        else:
            xr_cube = self.bolvanka_full.copy()
        
        xr_cube['data'] = (xr_cube.dims, self.clustered_volume)

        results, X, Y, Z = extractor.extract_and_create_map(
                          cube=xr_cube,
                          data_var='data',
                          output_nc_file= None, sec_name=section_conf['name'], 
                          sec_val=section_conf['val'])

        self.ari_score = self.calculate_ari(facies_mask, Z)

        fig, ax = extractor.plot_cluster_map_as_image(X, Y, Z, 
                                                      title=f'Cluster Map, ar_score: {self.ari_score:.3f}')
        self.run_study.upload(f'reflect_horiz_trial{self.trial.number}_clust{class_num}.pdf', fig)

        plot_conf = {}
        plot_conf['surfaces'] = self.surf_plot
        plot_conf[section_conf['name']] = section_conf['val']
        fig, ax = save_pic_only_horizon(xr_cube, plot_conf)
        self.run_study.upload(f'cross_sec_trial{self.trial.number}_clust{class_num}.pdf', fig)
        return self.ari_score

    def read_bolvankas(self):
        bolv_full_path = self.config['data_paths']['full_cube_templ']
        down_cube_templ_path = self.config['data_paths']['down_cube_templ']
        self.bolvanka_full = xr.load_dataset(bolv_full_path)
        self.bolvanka_down = xr.load_dataset(down_cube_templ_path)
    
    def read_surf_to_plot(self):
        # needed_secs = {'xline': 2693,
        #                'surfaces':{
        #                         'upper': upper_surf,
        #                         'lower': lower_surf,
        #                         'reflecting_horizon': reflect_horizon
        #                         }
        #                }

        df_fold = self.config['data_paths']['surf_data_plot']
        self.surf_plot = {}
        for df_name in os.listdir(df_fold):
            name = df_name.split('.')[0]
            df = pd.read_csv(os.path.join(df_fold, df_name))
            self.surf_plot[name] = df

    def calculate_ari(self, ground_truth, prediction):
        """
        Calculate ARI only on valid regions (where ground_truth != -1)
        """
        # Flatten and filter valid points
        valid_mask = (ground_truth != -1) & (~np.isnan(ground_truth))
        gt_valid = ground_truth[valid_mask].flatten()
        pred_valid = prediction[valid_mask].flatten()
        
        # Remove any remaining invalid values
        valid_idx = ~np.isnan(pred_valid)
        gt_valid = gt_valid[valid_idx]
        pred_valid = pred_valid[valid_idx]
        
        return adjusted_rand_score(gt_valid, pred_valid)

    def get_section_picture(self, isec_n=1200, xsec_n=2500):
        isec_n = isec_n // 4 if self.clustered_volume.shape[0] < 1000 else isec_n
        xsec_n = xsec_n // 4 if self.clustered_volume.shape[1] < 2000 else xsec_n
        fig, ax = plt.subplots(2, 1, figsize=(10,6))
        n_clust = self.cluster_num + 1
        # Add global title to the figure
        fig.suptitle("Cube sections", fontsize=16, fontweight='bold')
        cmap = plt.cm.get_cmap('tab10', n_clust)
        isec = self.clustered_volume[isec_n]
        xsec = self.clustered_volume[:, xsec_n, :]
        
        im0 = ax[0].imshow(isec.T, cmap=cmap)
        im1 = ax[1].imshow(xsec.T, cmap=cmap)
        
        # Add small titles for each subplot
        ax[0].set_title(f"Iline section: {isec_n}")
        ax[1].set_title(f"Xline section: {xsec_n}")
        
        # Add colorbars with smaller size
        cbar1 = fig.colorbar(im0, ax=ax[0], fraction=0.03, pad=0.04, 
                             shrink=0.5,  ticks=np.arange(n_clust))
        cbar2 = fig.colorbar(im1, ax=ax[1], fraction=0.03, pad=0.04, 
                             shrink=0.5,  ticks=np.arange(n_clust))
        
        fig.tight_layout()
        self.run_study.upload(f"cube_sects_clust_n_{self.cluster_num}_{self.trial.number}.pdf", fig)
        
        # os.makedirs(f"{self.result_fold}/tsne", exist_ok=True)
        # plt.savefig(f"{self.result_fold}/tsne/latent_tsne_trial_{self.trial.number}.png", 
        #            dpi=150, bbox_inches='tight')
        
        # print("t-SNE visualization saved and logged to Neptune")
        plt.close(fig)

    def run_complete_inference(self):
        
        self.get_latent_vecs()
        fst_clust_num = self.clust_options[0]
        best_loop = -np.inf
        
        for clust_n in self.clust_options:
            self.cluster_num = clust_n
            
            self.latent_space_clust()
            if len(np.unique(self.cluster_labels)) == 1:
                print('Only one unique class was clustered!')
                continue
            # stats = self.get_cluster_statistics()
            # print(f"Cluster statistics: {stats}")
            
            self.create_clustered_volume()
            ar_score = self.clust_cube_process(clust_n)
            best_loop = ar_score if ar_score > best_loop else best_loop
            best_metric_path = os.path.join(self.result_fold, 'best_metric.txt')
            best_global = read_or_create_metric(best_metric_path,
                                                ar_score)
            # if (metric > value) and (optim_direction=='maximize'):
            #         f.write(str(metric))
            #     elif (metric < value) and (optim_direction=='minimize'):
            #         f.write(str(metric))
            #     print(f"Read value {value} from {filepath}")
            #     return value
            if (ar_score >= best_global) and (self.config['direction'] =='maximize'):
                with open(os.path.join(self.result_fold, 'best_metric.txt'), 'w'):
                    f.write(str(ar_score))
                # First trial - always save as baseline
                self.tsne_visual()
                self.save_clust_cube()

        return best_loop
            
            # return self.clustered_volume