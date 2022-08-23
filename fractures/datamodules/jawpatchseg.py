from itertools import product
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
from scipy import ndimage
import torch
from torchtyping import TensorType

from fractures.data.datasets import JawFracDataset
from fractures.data.transforms import (
    Compose,
    IntensityAsFeatures,
    ToTensor,
    IntersectionOverUnion,
    CropCenters,
    BoneCenters,
    BoundingBoxes,
)
from fractures.datamodules.jawseg import JawSegDataModule


class JawPatchSegDataModule(JawSegDataModule):

    def __init__(
        self,
        crop_size: int,
        stride: int,
        bone_intensity_thresh: float,
        bone_volume_thresh: float,
        pos_iou_thresh: float,
        **dm_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(**dm_cfg)

        self.default_transforms = Compose(
            IntensityAsFeatures(),
            # CropCenters(crop_size=64, stride=32),
            # BoneCenters(),
            # BoundingBoxes(),
            # IntersectionOverUnion(),
            ToTensor(),
        )

        self.crop_size = crop_size
        self.stride = stride
        self.intensity_thresh = bone_intensity_thresh
        self.volume_thresh = bone_volume_thresh
        self.iou_thresh = pos_iou_thresh

    def setup(self, stage: Optional[str]=None) -> None:
        if stage is None or stage == 'fit':
            files = self._files('fit')
            train_files, val_files = self._split(files)

            self.train_dataset = JawFracDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                transform=self.default_transforms,
            )
            self.val_dataset = JawFracDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                transform=self.default_transforms,
            )

    @property
    def num_channels(self) -> int:
        return 1

    def crop_slices(
        self,
        features: TensorType['C', 'H', 'W', 'D', torch.float32],
    ) -> List[Tuple[slice, slice, slice, slice]]:
        xyz_coords = []
        for dim in features.shape[1:]:
            start_coords = np.arange(0, dim - self.crop_size // 2, self.stride)
            start_coords = np.clip(start_coords, 0, dim - self.crop_size)

            xyz_coords.append(start_coords)

        crop_coords = np.stack(list(product(*xyz_coords)))
        slice_coords = np.dstack((crop_coords, crop_coords + self.crop_size))
        slices = [tuple(slice(*c) for c in coords) for coords in slice_coords]
        slices = [(slice(None),) + slices for slices in slices]

        return slices

    def bone_crop_slices(
        self,
        features: TensorType['C', 'H', 'W', 'D', torch.float32],
        slices: List[Tuple[slice, slice, slice, slice]],
    ) -> List[Tuple[slice, slice, slice, slice]]:
        bone_slices = []
        for slices in slices:
            patch_bone = features[slices] >= self.intensity_thresh
            if patch_bone.float().mean() >= self.volume_thresh:
                bone_slices.append(slices)

        return bone_slices

    def bbox_ious(
        self,
        labels: TensorType['H', 'W', 'D', torch.float32],
        slices: List[Tuple[slice, slice, slice, slice]],
    ) -> TensorType['P', torch.float32]:
        bbox_slices = []
        for obj in ndimage.find_objects(labels):
            bbox_slices.append((
                slice(obj[0].start, obj[0].stop),
                slice(obj[1].start, obj[1].stop),
                slice(obj[2].start, obj[2].stop),
            ))

        filled = torch.zeros(labels.shape)
        for bbox_slices in bbox_slices:
            filled[bbox_slices] = 1

        ious = torch.zeros(len(slices))
        for i, slices in enumerate(slices):
            slices = slices[1:]
            ious[i] = torch.any(labels[slices]) * filled[slices].mean()

        return ious        

    def crop_patches(
        self,
        features: TensorType['C', 'H', 'W', 'D', torch.float32],
        labels: TensorType['H', 'W', 'D', torch.float32],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        TensorType['P', torch.float32],
    ]:
        slices = self.crop_slices(features)
        slices = self.bone_crop_slices(features, slices)

        ious = self.bbox_ious(labels, slices)
        pos_idxs = (ious >= self.iou_thresh).nonzero()[:, 0]
        neg_idxs = (ious < self.iou_thresh).nonzero()[:, 0]
        neg_idxs = neg_idxs[torch.randperm(neg_idxs.shape[0])]
        neg_idxs = neg_idxs[:pos_idxs.shape[0]]
        idxs = torch.cat((pos_idxs, neg_idxs))

        slices = [slices[i] for i in idxs]
        patches = torch.stack([features[slices] for slices in slices])
        ious = ious[idxs]
        
        return patches, ious

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'size', 'size', 'size', torch.float32],
        TensorType['P', torch.float32],
    ]:
        patches = torch.zeros(0, self.num_channels, *(self.crop_size,)*3)
        ious = torch.zeros(0)
        for data_dict in batch:
            batch_patches, batch_ious = self.crop_patches(
                data_dict['intensities'], data_dict['labels'],
            )
            
            patches = torch.cat((patches, batch_patches))
            ious = torch.cat((ious, batch_ious))

        return patches, ious
