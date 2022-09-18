from typing import Any, Dict, List, Tuple

import nibabel
import numpy as np
import pytorch_lightning as pl
from scipy import interpolate, ndimage
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import F1Score
from torchtyping import TensorType
from torch_scatter import scatter_mean

import mandibles.nn as nn
from miccai.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)


def patches_subgrid(
    patch_idxs: TensorType['P', 3, 2, torch.int64],
) -> Tuple[
    TensorType['P', 3, torch.int64],
    List[TensorType['D', torch.float32]],
    Tuple[int, int, int],
]:
    # determine subgrid covered by patches
    subgrid_idxs, subgrid_points = [], []
    for dim_patch_idxs in patch_idxs.permute(1, 0, 2):
        unique, inverse = torch.unique(
            input=dim_patch_idxs, return_inverse=True, dim=0,
        )
        subgrid_idxs.append(inverse)

        centers = unique.float().mean(dim=1)
        subgrid_points.append(centers)

    subgrid_idxs = torch.column_stack(subgrid_idxs)
    subgrid_shape = tuple(p.shape[0] for p in subgrid_points)

    return subgrid_idxs, subgrid_points, subgrid_shape


def batch_forward(
    model: torch.nn.Module,
    features: TensorType['C', 'D', 'H', 'W', torch.float32],
    patch_slices: List[Tuple[slice, slice, slice]],
    batch_size: int=24,
):
    out = []
    for i in range(0, len(patch_slices), batch_size):
        # extract batch of patches from features volume
        x = []
        for slices in patch_slices[i:i + batch_size]:
            slices = (slice(None),) + slices
            patch = features[slices]
            x.append(patch)

        # predict relative position and segmentation of patches
        x = torch.stack(x)
        pred = model(x)
        out.append(pred)

    return out


def project_patches(
    model: torch.nn.Module,
    features: TensorType['C', 'D', 'H', 'W', torch.float32],
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    subgrid_idxs: TensorType['P', 3, torch.int64],
    subgrid_shape: Tuple[int, int, int],
) -> Tuple[
    TensorType['d', 'h', 'w', 3, torch.float32],
    TensorType['D', 'H', 'W', torch.float32],
]:
    # convert patch indices to patch slices
    patch_slices = []
    for patch_idxs in patch_idxs:
        slices = tuple(slice(start, stop) for start, stop in patch_idxs)
        patch_slices.append(slices)

    # do model processing in batches
    out = batch_forward(model, features, patch_slices)
    coords, seg = map(lambda t: torch.cat(t), zip(*out))

    # put position prediction in subgrid
    out_coords = torch.zeros(subgrid_shape + (3,)).to(features)
    index_arrays = tuple(subgrid_idxs.T)
    out_coords[index_arrays] = coords

    # compute maximum of overlapping predictions
    out_seg = torch.full_like(features[0], -float('inf'))
    for slices, seg in zip(patch_slices, seg):
        out_seg[slices] = torch.maximum(out_seg[slices], seg)

    return out_coords, out_seg


def interpolate_positions(
    coords: TensorType['d', 'h', 'w', 3, torch.float32],
    subgrid_points: List[TensorType['D', 3, torch.float32]],
    out_shape: Tuple[int, int, int],
) -> TensorType['D', 'H', 'W', 3, torch.float32]:
    # compute coordinates of output voxels
    out_points = [np.arange(dim_size) + 0.5 for dim_size in out_shape]
    out_points = np.meshgrid(*out_points, indexing='ij')
    out_points = np.stack(out_points, axis=3)

    # interpolate subgrid values to output voxels
    out_coords = interpolate.interpn(
        points=[p.cpu() for p in subgrid_points],
        values=coords.cpu(),
        xi=out_points,
        bounds_error=False,
        fill_value=None,
    )

    return torch.from_numpy(out_coords).to(coords)


def filter_connected_components(
    coords: TensorType['D', 'H', 'W', 3, torch.float32],
    seg: TensorType['D', 'H', 'W', torch.float32],
    conf_thresh: float=0.6,
) -> TensorType['D', 'H', 'W', torch.bool]:
    # determine connected components in volume
    probs = torch.sigmoid(seg)
    labels = (probs >= conf_thresh).long()
    component_idxs, _ = ndimage.label(
        input=labels.cpu(),
        structure=ndimage.generate_binary_structure(3, 1),
    )
    component_idxs = torch.from_numpy(component_idxs).to(labels)

    # determine components comprising at least 20k voxels
    _, component_idxs, component_counts = torch.unique(
        component_idxs, return_inverse=True, return_counts=True,
    )
    count_mask = component_counts >= 20_000

    # determine components with mean confidence at least 0.70
    component_probs = scatter_mean(
        src=probs.flatten(),
        index=component_idxs.flatten(),
    )
    prob_mask = component_probs >= 0.6

    # determine components within two variance of centroid
    coords[..., 0] -= 1  # left-right symmetry
    component_coords = scatter_mean(
        src=coords.reshape(-1, 3),
        index=component_idxs.flatten(),
        dim=0,
    )
    component_dists = torch.sum(component_coords ** 2, dim=1)
    dist_mask = component_dists < 2

    # project masks back to volume
    component_mask = count_mask & prob_mask & dist_mask
    volume_mask = component_mask[component_idxs]

    return volume_mask


def fill_source_volume(
    volume_mask: TensorType['D', 'H', 'W', torch.bool],
    affine: TensorType[4, 4, torch.float32],
    shape: TensorType[3, torch.int64],
) -> TensorType['D', 'H', 'W', torch.bool]:
    # compute voxel indices into source volume
    voxels = volume_mask.nonzero().float()
    hom_voxels = torch.column_stack((voxels, torch.ones_like(voxels[:, 0])))

    orig_voxels = torch.einsum(
        'ij,kj->ki', torch.linalg.inv(affine), hom_voxels,
    )
    orig_voxels = orig_voxels[:, :3].round().long()

    # initialize empty volume and fill with binary segmentation
    out = torch.zeros(shape.tolist(), dtype=torch.bool)
    out[tuple(orig_voxels.T)] = True

    # dilate volume to fill empty space between foreground voxels
    out = ndimage.binary_dilation(
        input=out,
        structure=ndimage.generate_binary_structure(3, 2),
        iterations=3,
    )

    return torch.from_numpy(out).to(volume_mask)


class MandiblePatchSegModule(pl.LightningModule):

    def __init__(
        self,
        lr: float,
        epochs: int,
        warmup_epochs: int,
        weight_decay: float,
        **model_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model = nn.MandibleNet(**model_cfg)
        self.criterion = nn.MandibleLoss(alpha=0.25, gamma=2.0)
        self.f1 = F1Score(num_classes=2, average='macro')

        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

    def forward(
        self,
        x: TensorType['P', 'C', 'size', 'size', 'size', torch.float32],       
    ) -> Tuple[
        TensorType['P', 3, torch.float32],
        TensorType['P', 'size', 'size', 'size', torch.float32],
    ]:
        coords, seg = self.model(x)

        return coords, seg

    def training_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 3, torch.float32],
                TensorType['P', 'size', 'size', 'size', torch.int64],
            ],
        ],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, y = batch

        coords, seg = self(x)

        loss, log_dict = self.criterion(coords, seg, y)

        self.log_dict({
            k.replace('/', '/train_' if k[-1] != '/' else '/train'): v
            for k, v in log_dict.items()
        })

        return loss

    def validation_step(
        self,
        batch: Tuple[
            TensorType['P', 'C', 'size', 'size', 'size', torch.float32],
            Tuple[
                TensorType['P', 3, torch.float32],
                TensorType['P', 'size', 'size', 'size', torch.int64],
            ],
        ],
        batch_idx: int,
    ) -> None:
        x, y = batch

        coords, seg = self(x)

        _, log_dict = self.criterion(coords, seg, y)
        self.f1((seg >= 0).long().flatten(), y[1].flatten())

        self.log_dict({
            **{
                k.replace('/', '/val_' if k[-1] != '/' else '/val'): v
                for k, v in log_dict.items()
            },
            'f1/val': self.f1,
        })    

    def predict_volumes(
        self,
        features: TensorType['C', 'D', 'H', 'W', torch.float32],
        patch_idxs: TensorType['P', 3, 2, torch.int64],
    ) -> Tuple[
        TensorType['D', 'H', 'W', 3, torch.float32],
        TensorType['D', 'H', 'W', torch.float32],
    ]:
        # determine subgrid covered by patches
        subgrid = patches_subgrid(patch_idxs)
        idxs, points, shape = subgrid

        # combine overlapping voxel predictions
        coords, seg = project_patches(
            self.model, features, patch_idxs, idxs, shape,
        )

        # interpolate relative position predictions
        coords = interpolate_positions(coords, points, seg.shape)

        return coords, seg

    def test_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType['P', 3, torch.float32],
            TensorType['D', 'H', 'W', torch.int64],
        ],
        batch_idx: int,
    ) -> None:
        features, patch_idxs, patch_coords, labels = batch

        # predict binary segmentation
        coodrs, seg = self.predict_volumes(features, patch_idxs)
        mask = torch.sigmoid(seg) > 0.5

        # compute metrics
        self.f1(mask.long().flatten(), labels.flatten())
        
        # log metrics
        self.log('f1/test', self.f1)

    def predict_step(
        self,
        batch: Tuple[
            TensorType['C', 'D', 'H', 'W', torch.float32],
            TensorType['P', 3, 2, torch.int64],
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ) -> TensorType['D', 'H', 'W', torch.bool]:
        features, patch_idxs, affine, shape = batch

        # predict binary segmentation
        coords, seg = self.predict_volumes(features, patch_idxs)

        # remove small or low-confidence connected components
        volume_mask = filter_connected_components(coords, seg)

        # fill volume with original shape given foreground mask
        out = fill_source_volume(volume_mask, affine, shape)

        return out

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.01)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.0001)

        return [opt], [sch]
