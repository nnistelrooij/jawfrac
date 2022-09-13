from itertools import product
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from miccai.data.transforms import *


class NonNegativeCrop:

    def __init__(
        self,
        padding: Union[float, ArrayLike],
    ) -> None:
        self.padding = padding

    def __call__(
        self,
        intensities: NDArray[Any],
        spacing: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # find smallest bounding box that encapsulates all non-negative voxels
        slices = ndimage.find_objects(intensities >= 0)[0]

        # determine slices of bounding box after padding
        padded_slices = ()
        padding = np.ceil(self.padding / spacing).astype(int)
        for s, pad in zip(slices, padding):
            padded_slices += (slice(max(s.start - pad, 0), s.stop + pad),)
        
        # crop volumes given padded slices
        data_dict['intensities'] = intensities[padded_slices]
        if 'labels' in data_dict:
            data_dict['labels'] = data_dict['labels'][padded_slices]

        # determine affine transformation from source to crop
        affine = np.eye(4)
        affine[:3, 3] -= [s.start for s in padded_slices]

        data_dict['spacing'] = spacing
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    padding={self.padding},',
            ')',
        ])

    
class RegularSpacing:

    def __init__(
        self,
        spacing: Union[float, ArrayLike],
    ) -> None:
        if isinstance(spacing, float):
            spacing = [spacing]*3

        self.spacing = np.array(spacing)

    def __call__(
        self,
        spacing: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # compute how much bigger results should be
        zoom = spacing / self.spacing

        # interpolate the inputs to have regular voxel spacing
        data_dict['intensities'] = ndimage.zoom(data_dict['intensities'], zoom)
        if 'labels' in data_dict:
            data_dict['labels'] = ndimage.zoom(data_dict['labels'], zoom)
            data_dict['labels'] = (data_dict['labels'] >= 1).astype(np.int16)

        # determine affine transformation from input to result
        affine = np.eye(4)
        affine[np.diag_indices(3)] = zoom

        data_dict['spacing'] = self.spacing
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    spacing={self.spacing},',
            ')',
        ])


class NaturalHeadPositionOrient:

    def __call__(
        self,
        orientation: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # reorient volumes to standard basis
        for key in ['intensities', 'labels']:
            if key not in data_dict:
                continue

            shape = data_dict[key].shape
            data_dict[key] = nibabel.apply_orientation(
                arr=data_dict[key],
                ornt=orientation,
            )

        # determine affine transformation from input to result
        affine = nibabel.orientations.inv_ornt_aff(
            ornt=orientation,
            shape=shape,
        )
        affine = np.linalg.inv(affine)

        data_dict['orientation'] = nibabel.io_orientation(affine=np.eye(4))
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PatchIndices:

    def __init__(
        self,
        patch_size: int,
        stride: int,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # determine start indices of patches along each dimension
        start_idxs = []
        for dim_size in intensities.shape:
            dim_idxs = list(range(0, dim_size - self.patch_size, self.stride))
            dim_idxs = np.array(dim_idxs + [dim_size - self.patch_size])

            start_idxs.append(dim_idxs)

        # determine start and stop indices of patches in all dimensions
        start_idxs = np.stack(list(product(*start_idxs)))
        stop_idxs = start_idxs + self.patch_size
        patch_idxs = np.dstack((start_idxs, stop_idxs))

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = patch_idxs

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.patch_size},',
            f'    stride={self.stride},',
            ')',
        ])


class PositiveNegativePatchIndices:

    def __init__(
        self,
        volume_thresh: float=0.05,
    ) -> None:
        self.volume_thresh = volume_thresh

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        pos_patch_idxs, neg_patch_idxs = [], []
        for patch_idxs in data_dict['patch_idxs']:
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # determine if patch contains sufficient foreground voxels
            if patch.mean() >= self.volume_thresh:
                pos_patch_idxs.append(patch_idxs)
            else:
                neg_patch_idxs.append(patch_idxs)

        data_dict['pos_patch_idxs'] = np.stack(pos_patch_idxs)
        data_dict['neg_patch_idxs'] = np.stack(neg_patch_idxs)
        data_dict['labels'] = labels

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    pos_volume_thresh={self.volume_thresh},',
            ')',
        ])


class IntensityAsFeatures:

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], intensities[np.newaxis].astype(float)),
            )
        else:
            data_dict['features'] = intensities[np.newaxis].astype(float)

        data_dict['intensities'] = intensities

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PositiveNegativePatches:

    def __init__(
        self,
        max_patches: int,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max_pos_patches = max_patches // 2
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        features: NDArray[np.float64],
        labels: NDArray[Any],
        pos_patch_idxs: NDArray[np.int64],
        neg_patch_idxs: NDArray[np.int64],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # sample at most self.max_pos_patches positive patches from scan
        pos_patch_idxs = self.rng.permutation(pos_patch_idxs)
        pos_patch_idxs = pos_patch_idxs[:self.max_pos_patches]

        # sample as many negative patches as positive patches
        neg_patch_idxs = self.rng.permutation(neg_patch_idxs)
        neg_patch_idxs = neg_patch_idxs[:pos_patch_idxs.shape[0]]

        patch_idxs = np.concatenate((pos_patch_idxs, neg_patch_idxs))

        # crop patches from volumes
        feature_patches, label_patches = [], []
        for patch_idxs in patch_idxs:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            label_patch = labels[slices]
            label_patches.append(label_patch)

            slices = (slice(None),) + slices
            feature_patches.append(features[slices])

        # stack patches to make batch
        data_dict['features'] = np.stack(feature_patches)
        data_dict['labels'] = np.stack(label_patches)
        data_dict['pos_patch_idxs'] = pos_patch_idxs
        data_dict['neg_patch_idxs'] = neg_patch_idxs

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_patches={2 * self.max_pos_patches},',
            ')',
        ])
