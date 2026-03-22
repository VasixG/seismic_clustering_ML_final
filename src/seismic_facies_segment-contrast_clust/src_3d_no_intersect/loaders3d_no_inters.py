import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import torch.nn.functional as F


class SeismicPatchesDataset(Dataset):
    
    def __init__(
        self,
        attr_cubes: torch.Tensor,
        presence_mask: torch.Tensor,
        patch_size: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]] = None,
        min_mask_ratio: float = 0.5,
        region_mask=None
    ):
    
        self.attr_cubes = attr_cubes
        self.presence_mask = presence_mask
        self.patch_size = patch_size
        self._update_presence_mask(region_mask)
        self.stride = stride if stride is not None else patch_size
        self.min_mask_ratio = min_mask_ratio
        self.n_attrs, self.h, self.w, self.t = attr_cubes.shape
        
        self.valid_patches = self._extract_valid_patches()
        
        print(f"Dataset initialized with {len(self.valid_patches)} valid patches")
        print(f"Patch size: {patch_size}, Stride: {self.stride}")
    
    def _update_presence_mask(self, region_mask):
        if region_mask is None:
            return
        else:
            self.presence_mask = self.presence_mask * region_mask
    
    def _extract_valid_patches(self) -> List[Tuple[int, int, int]]:
    
        valid_patches = []
        
        n_z = (self.t - self.patch_size[2]) // self.stride[2] + 1
        n_y = (self.w - self.patch_size[1]) // self.stride[1] + 1
        n_x = (self.h - self.patch_size[0]) // self.stride[0] + 1
        
        total_patches = n_z * n_y * n_x
        print(f"Total possible patches: {total_patches}")
        
        
        for z in range(0, self.t - self.patch_size[2] + 1, self.stride[2]):
            for y in range(0, self.w - self.patch_size[1] + 1, self.stride[1]):
                for x in range(0, self.h - self.patch_size[0] + 1, self.stride[0]):
                    
                    mask_patch = self.presence_mask[
                        x:x + self.patch_size[0],
                        y:y + self.patch_size[1],
                        z:z + self.patch_size[2]
                    ]
                    
                    mask_ratio = mask_patch.float().mean().item()
                    
                    if mask_ratio >= self.min_mask_ratio:
                        valid_patches.append((x, y, z))
        
        return valid_patches
    
    def __len__(self) -> int:
        return len(self.valid_patches)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        
        x, y, z = self.valid_patches[idx]
        
        patch = self.attr_cubes[
            :,
            x:x + self.patch_size[0],
            y:y + self.patch_size[1],
            z:z + self.patch_size[2]
        ]
        
        return patch, None


class SeismicPatchesDsCategoric(Dataset):
    
    def __init__(
        self,
        attr_cubes: torch.Tensor,
        attr_categoric: torch.Tensor, 
        presence_mask: torch.Tensor,
        patch_size: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]] = None,
        min_mask_ratio: float = 0.5,
        region_mask=None
    ):
    
        self.attr_cubes = attr_cubes
        self.attr_categoric = attr_categoric
        assert self.attr_cubes.shape[1:] == self.attr_categoric.shape[1:], 'Categorical feat shape != continuous ones'
        self.presence_mask = presence_mask
        self._update_presence_mask(region_mask)
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.min_mask_ratio = min_mask_ratio
        

        self.n_attrs, self.h, self.w, self.t = attr_cubes.shape
        
        self.valid_patches = self._extract_valid_patches()
        
        print(f"Dataset initialized with {len(self.valid_patches)} valid patches")
        print(f"Patch size: {patch_size}, Stride: {self.stride}")
    
    def _update_presence_mask(self, region_mask):
        if region_mask is None:
            return
        else:
            self.presence_mask = self.presence_mask * region_mask


    def _extract_valid_patches(self) -> List[Tuple[int, int, int]]:
    
        valid_patches = []
        
        n_z = (self.t - self.patch_size[2]) // self.stride[2] + 1
        n_y = (self.w - self.patch_size[1]) // self.stride[1] + 1
        n_x = (self.h - self.patch_size[0]) // self.stride[0] + 1
        
        total_patches = n_z * n_y * n_x
        print(f"Total possible patches: {total_patches}")
        
        
        for z in range(0, self.t - self.patch_size[2] + 1, self.stride[2]):
            for y in range(0, self.w - self.patch_size[1] + 1, self.stride[1]):
                for x in range(0, self.h - self.patch_size[0] + 1, self.stride[0]):
                    
                    mask_patch = self.presence_mask[
                        x:x + self.patch_size[0],
                        y:y + self.patch_size[1],
                        z:z + self.patch_size[2]
                    ]
                    
                    mask_ratio = mask_patch.float().mean().item()
                    
                    if mask_ratio >= self.min_mask_ratio:
                        valid_patches.append((x, y, z))
        
        return valid_patches
    
    def __len__(self) -> int:
        return len(self.valid_patches)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        
        x, y, z = self.valid_patches[idx]
        
        patch = self.attr_cubes[
            :,
            x:x + self.patch_size[0],
            y:y + self.patch_size[1],
            z:z + self.patch_size[2]
        ]

        categoric_patch = self.attr_categoric[
            :,
            x:x + self.patch_size[0],
            y:y + self.patch_size[1],
            z:z + self.patch_size[2]
        ].long() 
        
        return patch, categoric_patch