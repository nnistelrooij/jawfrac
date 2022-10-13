from typing import List, Tuple

import numpy as np
from scipy import interpolate, ndimage
import torch
from torchtyping import TensorType


def batch_forward(
    model: torch.nn.Module,
    features: TensorType['C', 'D', 'H', 'W', torch.float32],
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    x_axis_flip: bool=True,
    batch_size: int=12,
):
    # convert patch indices to patch slices
    patch_slices = []
    for patch_idxs in patch_idxs:
        slices = tuple(slice(start, stop) for start, stop in patch_idxs)
        patch_slices.append((slice(None),) + slices)

    batches = []
    batch_size //= 1 + x_axis_flip
    for i in range(0, len(patch_slices), batch_size):
        # extract batch of patches from features volume
        x = []
        for slices in patch_slices[i:i + batch_size]:
            patch = features[slices]
            x.append(patch)
            if x_axis_flip: x.append(patch.fliplr())

        # predict relative position and segmentation of patches
        x = torch.stack(x)
        batch = model(x)
        batches.append(batch)

    # get tuple of concatenated output batches
    if isinstance(batches[0], torch.Tensor):
        batches = (batches,)
    out = tuple(torch.cat(batches) for batches in zip(*batches))

    # return simplest data structure
    if not x_axis_flip:
        return out[0] if len(out) == 1 else out

    # take mean of predictions from same patches
    for i, batches in enumerate(out):
        batches = torch.stack((
            batches[::2],
            batches[1::2].fliplr() if batches.dim() == 4 else batches[1::2],
        ))
        preds = torch.mean(batches, dim=0)
        out = out[:i] + (preds,) + out[i + 1:]

    # return simplest data structure
    return out[0] if len(out) == 1 else out


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


def interpolate_sparse_predictions(
    pred: TensorType['d', 'h', 'w', '...', torch.float32],
    subgrid_points: List[TensorType['dim', 3, torch.float32]],
    out_shape: torch.Size,
    scale: int=4,
) -> TensorType['D', 'H', 'W', '...', torch.float32]:
    # compute coordinates of output voxels
    points_down = []
    for dim in out_shape:
        points = np.linspace(0, dim - dim % scale, num=dim // scale + 1)
        points_down.append(points + scale * 0.5)
    points_down = np.meshgrid(*points_down, indexing='ij')
    points_down = np.stack(points_down, axis=3)

    # interpolate and extrapolate subgrid values to output voxels
    out_down = interpolate.interpn(
        points=[p.cpu() for p in subgrid_points],
        values=pred.cpu(),
        xi=points_down,
        bounds_error=False,
        fill_value=None,
    )
    out_down = torch.from_numpy(out_down).to(pred)

    # repeat values to scale up interpolated predictions
    out_interp = out_down.tile(*(scale,)*3, *(1,)*(pred.dim() - 3))
    out_interp = out_interp[tuple(slice(None, dim) for dim in out_shape)]

    return out_interp


def aggregate_sparse_predictions(
    pred: TensorType['P', '...', torch.float32],
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    out_shape: torch.Size,
) -> TensorType['D', 'H', 'W', '...', torch.float32]:
    subgrid_idxs, subgrid_points, subgrid_shape = patches_subgrid(patch_idxs)

    out = torch.zeros(subgrid_shape + pred.shape[1:])
    out = out.to(pred) + pred.amin(dim=0)
    index_arrays = tuple(subgrid_idxs.T)
    out[index_arrays] = pred

    out = interpolate_sparse_predictions(
        out, subgrid_points, out_shape,
    )

    return out


def aggregate_dense_predictions(
    pred: TensorType['P', 'size', 'size', 'size', torch.float32],
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    out_shape: torch.Size,
    mode='max',
) -> TensorType['D', 'H', 'W', torch.float32]:
    # convert patch indices to patch slices
    patch_slices = []
    for patch_idxs in patch_idxs:
        slices = tuple(slice(start, stop) for start, stop in patch_idxs)
        patch_slices.append(slices)

    if mode == 'max':
        # compute maximum of overlapping predictions
        out = torch.zeros(out_shape)
        out = out.to(pred) + pred.amin()
        for slices, pred in zip(patch_slices, pred):
            out[slices] = torch.maximum(out[slices], pred)
    elif mode == 'mean':
        # compute mean of overlapping predictions
        out = torch.zeros(out_shape).to(pred)
        for slices, pred in zip(patch_slices, pred):
            out[slices] += pred
    else: raise ValueError(f'Mode not recognized: {mode}.')

    return out


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
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=1,
    )

    return torch.from_numpy(out).to(volume_mask)
