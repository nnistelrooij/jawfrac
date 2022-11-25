from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate, ndimage
import torch
from torchtyping import TensorType
from tqdm import trange


def half_stride_patch_idxs(
    patch_idxs: TensorType['d', 'h', 'w', 3, 2, torch.int64],
) -> Tuple[
    TensorType['D', torch.int64],
    TensorType['D', torch.int64],
    TensorType['D', torch.int64],
]:
    dimx, dimy, dimz = patch_idxs.shape[:3]

    return torch.meshgrid(
        torch.arange(0, dimx + 1, 2).clip(0, dimx - 1),
        torch.arange(0, dimy + 1, 2).clip(0, dimy - 1),
        torch.arange(0, dimz + 1, 2).clip(0, dimz - 1),
        indexing='ij',
    )


def batch_forward(
    model: torch.nn.Module,
    features: TensorType['C', 'D', 'H', 'W', torch.float32],
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    x_axis_flip: bool,
    batch_size: int,
):
    # convert patch indices to patch slices
    patch_slices = []
    for patch_idxs in patch_idxs:
        slices = tuple(slice(start, stop) for start, stop in patch_idxs)
        patch_slices.append((slice(None),) + slices)

    batch_size //= 1 + x_axis_flip
    for i in trange(0, len(patch_slices), batch_size, leave=False):
        # extract batch of patches from features volume
        x = []
        for slices in patch_slices[i:i + batch_size]:
            patch = features[slices]
            x.append(patch)
            if x_axis_flip: x.append(patch.fliplr())

        # predict relative position and segmentation of patches
        x = torch.stack(x)
        batch = model(x)

        if not x_axis_flip:
            yield batch
            continue

        if isinstance(batch, torch.Tensor):
            batch = (batch,)

        for i, pred in enumerate(batch):
            pred = torch.stack((
                pred[::2],
                pred[1::2].fliplr() if pred.dim() == 4 else pred[1::2],
            ))
            batch = batch[:i] + (pred.mean(dim=0),) + batch[i + 1:]

        yield batch if len(batch) > 1 else batch[0]


def patches_subgrid(
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    mask: Optional[TensorType['P', torch.bool]],
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

    if mask is not None:
        subgrid_idxs = subgrid_idxs[mask]

    return subgrid_idxs, subgrid_points, subgrid_shape


def interpolate_sparse_predictions(
    pred: TensorType['d', 'h', 'w', '...', torch.float32],
    subgrid_points: List[TensorType['dim', 3, torch.float32]],
    out_shape: torch.Size,
    scale: int=2,
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
    out_interp = torch.from_numpy(out_down).to(pred)

    # repeat values to scale up interpolated predictions
    for dim in range(3):
        out_interp = out_interp.repeat_interleave(scale, dim=dim)
    out_interp = out_interp[tuple(slice(None, dim) for dim in out_shape)]

    return out_interp


def aggregate_sparse_predictions(
    pred_shape: torch.Size,
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    mask: Optional[TensorType['P', torch.bool]]=None,
    fill_value: Optional[float]=None,
    out_shape: Optional[torch.Size]=None,
) -> TensorType['D', 'H', 'W', '...', torch.float32]:
    subgrid = patches_subgrid(patch_idxs, mask)
    subgrid_idxs, subgrid_points, subgrid_shape = subgrid

    if fill_value is None:
        out = torch.empty(subgrid_shape + pred_shape)
    else:
        out = torch.full(subgrid_shape + pred_shape, float(fill_value))

    out = out.to(patch_idxs.device)
    pred = torch.empty(pred_shape).to(patch_idxs.device)
    yield pred

    for idxs in subgrid_idxs:
        out[tuple(idxs)] = pred
        yield

    if out_shape is None:
        yield out

    out = interpolate_sparse_predictions(
        out, subgrid_points, out_shape,
    )

    yield out


def aggregate_dense_predictions(
    patch_idxs: TensorType['P', 3, 2, torch.int64],
    out_shape: torch.Size,
    mode='max',
) -> TensorType['D', 'H', 'W', torch.float32]:
    pred_shape = tuple(patch_idxs[0, :, 1] - patch_idxs[0, :, 0])

    # convert patch indices to patch slices
    patch_slices = []
    for patch_idxs in patch_idxs:
        slices = tuple(slice(start, stop) for start, stop in patch_idxs)
        patch_slices.append(slices)

    if mode == 'max':
        # compute maximum of overlapping predictions
        out = torch.full(out_shape, float('-inf'), device=patch_idxs.device)
        pred = torch.empty(pred_shape, device=patch_idxs.device)
        yield pred

        for slices in patch_slices:
            out[slices] = torch.maximum(out[slices], pred)
            yield
    elif mode == 'mean':
        # compute mean of overlapping predictions
        out = torch.zeros(out_shape, device=patch_idxs.device)
        pred = torch.empty(pred_shape, device=patch_idxs.device)
        yield pred

        for slices in patch_slices:
            out[slices] += pred
            yield
    else:
        raise ValueError(f'Mode not recognized: {mode}.')

    yield out


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
    orig_voxels = orig_voxels.clip(torch.zeros_like(shape), shape - 1)
    orig_voxels = orig_voxels.cpu().numpy()

    # initialize empty volume and fill with binary segmentation
    out = np.zeros(shape.tolist(), dtype=bool)
    out[tuple(orig_voxels.T)] = True

    # dilate volume to fill empty space between foreground voxels
    spacing = affine[:, :3].sum(dim=0).abs()
    for i, dim in enumerate(spacing):
        if dim > 1 or torch.isclose(dim, torch.ones_like(dim)):
            continue

        iterations = torch.ceil(1 / dim).long().item() - 1

        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1 - (i == 0), 1 - (i == 1), 1 - (i == 2)] = True
        structure[1, 1, 1] = True
        structure[1 + (i == 0), 1 + (i == 1), 1 + (i == 2)] = True

        out = ndimage.binary_dilation(
            input=out,
            structure=structure,
            iterations=iterations,
        )

    return torch.from_numpy(out).to(volume_mask)
