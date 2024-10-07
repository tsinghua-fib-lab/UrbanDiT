"""
Originally inspired by impl at https://github.com/microsoft/unilm/tree/master/beit

Modified by Haoyu Lu, for generating the spatial-temporal masked position for video diffusion transformer
"""

import random
import math
import numpy as np
import torch



class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, 
            min_aspect=0.3,):

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches

        max_aspect = 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape())
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class VideoMaskGenerator:
    def __init__(self, input_size, spatial_mask_ratio=0.5, pred_len=12, his_len = 12):
        self.length, self.height, self.width = input_size

        self.spatial_generator = MaskingGenerator((self.height, self.width), spatial_mask_ratio * self.height * self.width)
        
        # idx = 0 Predict
        self.predict_given_frame_length = pred_len

        # idx = 1 Backward
        self.backward_given_frame_length = his_len

        # idx = 2 Interpreation
        self.interpreation_step = 2

        # idx = 5 MLM ratio
        self.mlm_ratio = 0.5

    def __repr__(self):
        repr_str = "Generator(%d, %d, %d)" % (
            self.length, self.height, self.width)
        return repr_str

    def get_shape(self):
        return self.length, self.height, self.width

    def spatial_mask(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mask = np.random.sample([self.height, self.width])
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        mask = np.tile(mask, (self.length, 1, 1))

        return mask

    def spatial_temporal_mask(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mask = np.random.sample([self.length, self.height, self.width])
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask

    def super_mask(self):
        mask = np.ones(shape=self.get_shape())
        return mask

    def temporal_mask(self, idx=0, seed=None):
        mask = np.zeros(shape=self.get_shape())
        # Predict
        if idx == 0:
            mask[self.predict_given_frame_length:] = 1
        # Backward
        elif idx == 1:
            mask[:-self.backward_given_frame_length] = 1
        # Interpreation
        elif idx == 2:
            mask = np.ones(shape=self.get_shape())
            mask[::self.interpreation_step] = 0
        elif idx == 3:
            frame_idx = random.randint(0, mask.shape[0]-1)
            mask = np.ones(shape=self.get_shape())
            mask[frame_idx] = 0

    def __call__(self, batch_size=1, device=None, idx=0, seed=None):
        if idx >= 0:
            if idx < 4: # 0, 1, 2, 3
                mask = self.temporal_mask(idx, seed)
            elif idx == 4: 
                mask = self.spatial_mask(seed)
            elif idx == 5:
                mask = self.spatial_temporal_mask(seed)
            return torch.tensor(mask,device=device).unsqueeze(0).repeat(batch_size,1,1,1).int()