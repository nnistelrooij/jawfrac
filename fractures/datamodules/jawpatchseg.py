from typing import Any, Dict, List, Optional, Tuple
import torch
from torchtyping import TensorType

from fractures.data.datasets import JawFracDataset
import fractures.data.transforms as T
from fractures.datamodules.jawfrac import JawFracDataModule


class JawFracPatchDataModule(JawFracDataModule):

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

        self.default_transforms = T.Compose(
            T.CropCenters(crop_size=crop_size, stride=32),
            T.BoneCenters(intensity_thresh=500),
            T.BoundingBoxes(),
            T.IntersectionOverUnion(),
            T.IntensityAsFeatures(
                level=450.0, width=1100.0,
                low=-1.0, high=1.0,
            ),
            T.ToTensor(),
        )

        self.crop_size = (crop_size,)*3
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
                **self.dataset_cfg,
            )
            self.val_dataset = JawFracDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

        if stage is None or stage == 'predict':
            files = self._files('predict')

            self.predict_dataset = JawFracDataset(
                stage='predict',
                root=self.root,
                files=files,
                transform=self.default_transforms,
                **self.dataset_cfg,
            )

    @property
    def num_channels(self) -> int:
        return 1      

    def crop_patches(
        self,    
        features: TensorType['C', 'H', 'W', 'D', torch.float32],
        slices: TensorType['S', 3, 2, torch.int64],
        ious: TensorType['S', torch.float32],
        **kwargs,
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        TensorType['P', torch.float32],
    ]:
        # determine balanced sample of patches
        pos_idxs = (ious >= self.iou_thresh).nonzero()[:, 0]
        neg_idxs = (ious < self.iou_thresh).nonzero()[:, 0]
        neg_idxs = neg_idxs[torch.randperm(neg_idxs.shape[0])]
        neg_idxs = neg_idxs[:pos_idxs.shape[0]]
        idxs = torch.cat((pos_idxs, neg_idxs))

        # crop patches from volume and stack
        patches = torch.zeros(0, self.num_channels, *self.crop_size)
        for slices in slices[idxs].numpy():
            crop = (slice(None),) + tuple(map(lambda s: slice(*s), slices))
            patches = torch.cat((patches, features[(None,) + crop]))

        ious = ious[idxs]
        
        return patches, ious

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
        TensorType['P', torch.float32],
    ]:
        patches = torch.zeros(0, self.num_channels, *self.crop_size)
        ious = torch.zeros(0)
        for data_dict in batch:
            batch_patches, batch_ious = self.crop_patches(**data_dict)
            
            patches = torch.cat((patches, batch_patches))
            ious = torch.cat((ious, batch_ious))

        return patches, ious
