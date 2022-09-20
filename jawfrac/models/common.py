from typing import List, Tuple

from scipy import ndimage
import torch
from torchtyping import TensorType


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
