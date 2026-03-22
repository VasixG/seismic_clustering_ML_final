# import xarray as xarray
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from loaders3d_no_inters import SeismicPatchesDataset
from loaders3d_no_inters import SeismicPatchesDsCategoric


class DataProcess3d:

    def __init__(self, config:dict, trial=None):

        self.config = config
        self.d_paths = self.config['data_paths']
        # self.ampl_cube = None
        self.presence_mask = None
        self.trial = trial
        self.full_dataset = None
        self.cube_stats = {'means': {},
                           'stds': {}}
        self.attr_cubes = []
        self.attr_categoric = []
        # if config['fixed_attribs']:
        self._load_data()
        
    
    def _load_data(self):
        
        mmap_mode = 'r+' if self.config['use_lazy_load'] else None
        print('Read cubes using mmap...') if mmap_mode else print(f'Read cubes entirely')    
        cubes_fold = self.d_paths['cubes_fold']
        # self.ampl_cube = np.load(os.path.join(cubes_fold,
        #                          self.d_paths['ampl_cube']), 
        #                          mmap_mode=mmap_mode)
        self.presence_mask = np.load(os.path.join(cubes_fold,
                                     self.d_paths['presence_mask']),  
                                     mmap_mode=mmap_mode)

        self.attr_names = self.d_paths['attrib_names']
        self.attr_categoric_names = []

        for attr in self.attr_names:
            attr_cube = np.load(os.path.join(cubes_fold, attr),
                                mmap_mode=mmap_mode)
            if 'categoric' in attr.lower():
                self.attr_categoric.append(attr_cube[None])
                self.attr_names.remove(attr)
                self.attr_categoric_names.append(attr)
            else:
                self.attr_cubes.append(attr_cube[None])
        
        if self.trial is not None:

            if 'use_twt' in self.config['parameters'].keys():
                use_twt = self.config['parameters']['use_twt']
                
                if_use_twt = self.trial.suggest_categorical(use_twt['name'],
                                                            use_twt['values'])
                if if_use_twt == 1:
                    twt_file = self.d_paths['twt_file']
                    twt_cube = np.load(os.path.join(cubes_fold, twt_file), mmap_mode=mmap_mode)
                    self.attr_names.append(twt_file)
                    self.attr_cubes.append(twt_cube[None])
            
            if 'use_spatial' in self.config['parameters'].keys():
                use_spatial = self.config['parameters']['use_spatial']
                
                if_use_spat = self.trial.suggest_categorical(use_spatial['name'],
                                                        use_spatial['values'])
                if if_use_spat == 1:
                    print('Attach cdp_x and cdp_y')
                    cdp_x_file = self.d_paths['cdp_x_file']
                    cdp_x_cube = np.load(os.path.join(cubes_fold, cdp_x_file), mmap_mode=mmap_mode)
                    self.attr_names.append(cdp_x_file)
                    self.attr_cubes.append(cdp_x_cube[None])
                    
                    cdp_y_file = self.d_paths['cdp_y_file']
                    cdp_y_cube = np.load(os.path.join(cubes_fold, cdp_y_file), mmap_mode=mmap_mode)
                    self.attr_names.append(cdp_y_file)
                    self.attr_cubes.append(cdp_y_cube[None])

        self.attr_cubes = np.concatenate(self.attr_cubes, axis=0)
        self.attr_categoric = np.concatenate(self.attr_categoric, axis=0) if self.attr_categoric else []
        print('Cubes has been read successfully!')
        print(f'Took {len(self.attr_names)} continuous attributes, \n Namely: {self.attr_names}')
        print(f'Took {len(self.attr_categoric)} categorical attributes, \n Namely: {self.attr_categoric_names}')
        print(f'Attribute shapes: {self.attr_cubes.shape}')
        self._check_data()
        self._normalize_data()

    def _check_data(self):
        print('Start checking data shapes')
        # ampl_cube_shape = self.ampl_cube.shape
        fst_cube_shape = self.attr_cubes[0].shape
        # print(f'An attribute shape: {fst_cube_shape}')
        shape_check = set([fst_cube_shape== x.shape 
                            for x in self.attr_cubes])
        assert len(shape_check) == 1, 'Cube shapes mismatch!'
        assert fst_cube_shape == self.presence_mask.shape, ('Attribute cube',
                                                                  'shape != presence mask shape')

    def _normalize_data(self):
        # print('Start normalization of cubes!')
        normal_strat = self.config['normalize_strat']
        if normal_strat == 'z_score':
            print('Z-score normalization strategy was taken!')
            normalize_fun = self._normalize_one_cube_z
        elif normal_strat == 'min_max':
            print('Min-max normalization strategy was taken!')
            normalize_fun = self._scale_a_cube_min_max
        else:
            raise ValueError(f'Normalization strategy is not defined: {normal_strat} given')
        # self.ampl_cube = self._normalize_one_cube(self.ampl_cube, 'ampl')
        for ind, attr_name in enumerate(self.attr_names):
            self.attr_cubes[ind] = normalize_fun(self.attr_cubes[ind],
                                                        attr_name)
            
            print(f'mean: {self.cube_stats["means"][attr_name]:.3f}, std: {self.cube_stats["stds"][attr_name]:.3f} for {attr_name}\n')

        print('Cubes has been normalized successfully!')
        # self.ampl_cube = torch.from_numpy(self.ampl_cube)
        self.attr_cubes = torch.from_numpy(self.attr_cubes)
        self.attr_categoric = (torch.from_numpy(self.attr_categoric) 
                                if not isinstance(self.attr_categoric, list) else None)
        self.presence_mask = torch.from_numpy(self.presence_mask)

    def _normalize_one_cube_z(self, cube, cube_name):
        # TODO: think how to normalize correctly, 
        # we take some zeros now in trainig ds
        # and we should derive normalizing constnts from the training ds

        data = cube[self.presence_mask]
        x_mean, x_std = data.mean(), data.std()
        cube = (cube - x_mean) / x_std
        self.cube_stats['means'][cube_name] = x_mean
        self.cube_stats['stds'][cube_name] = x_std
        return cube

    def _scale_a_cube_min_max(self, cube, cube_name):
        # TODO: think how to normalize correctly, 
        # we take some zeros now in trainig ds
        # and we should derive normalizing constnts from the training ds

        data = cube[self.presence_mask]
        x_min, x_max = data.min(), data.max()
        cube = (cube - x_min) / (x_max - x_min)

        print(f'min: {x_min:.3f}, max: {x_max:.3f} for {cube_name}')

        self.cube_stats['means'][cube_name] = data.mean()
        self.cube_stats['stds'][cube_name] = data.std()
        return cube

    def create_seismic_dataloaders(self,
                                dxy, dt,
                                training=True,
                                return_coords=False,
                                return_mask=False,
                                trial=None

                                ):
        
        self.get_seismic_ds(dxy, dt, training)
        if training:
            # Split into train/val
            train_split = self.config['train_test_ratio']
            dataset_size = len(self.full_dataset)
            train_size = int(train_split * dataset_size)
            test_size = dataset_size - train_size
            print(f"Train size: {train_size}, Test size: {test_size}")
            generator = torch.Generator()

            train_dataset, test_dataset = torch.utils.data.random_split(
                                            self.full_dataset, 
                                            [train_size, test_size],
                                            generator=generator
                                        )
            bs = self.config['train_bs']
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=bs,
                shuffle=True,
                drop_last=True
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False
            )

            return train_loader, test_loader
        else: 
            bs = self.config['inference_bs']
            
            inference_loader = torch.utils.data.DataLoader(
                                self.full_dataset,
                                batch_size=bs,  
                                shuffle=False,
                                # num_workers=4,
                                # pin_memory=True
                            )

            return inference_loader

    def get_seismic_ds(self, dxy, dt, training, region_mask=None):
        if (self.full_dataset is not None) and (not self.config['intersect']):
            # the same stride for training and inference
            return None
        min_mask_ratio = self.config['min_mask_ratio']
        
        patch_size = (dxy, dxy, dt)
        if self.config['intersect'] and training:
            stride = tuple(self.config['stride'])
            print('Configuration with intersection was picked!')
            print(f'Training stride: {stride}')
        elif self.config['intersect'] and not training:
            stride = self.config['inference_stride']
            print(f'Inference stride: {stride}')
        else:
            print('Configuration with no intersection picked')
            stride = None
        if self.attr_categoric is not None:
            print('Take dataset for categorical and continuous')
            self.full_dataset = SeismicPatchesDsCategoric(
                attr_cubes=self.attr_cubes,
                attr_categoric=self.attr_categoric,
                presence_mask=self.presence_mask,
                patch_size=patch_size,
                stride=stride,
                min_mask_ratio=min_mask_ratio,
                region_mask=region_mask 
            )
        else:
            print('Take dataset for only coninuous')

            self.full_dataset = SeismicPatchesDataset(
                attr_cubes=self.attr_cubes,
                presence_mask=self.presence_mask,
                patch_size=patch_size,
                stride=stride,
                min_mask_ratio=min_mask_ratio,
                region_mask=region_mask    
            )