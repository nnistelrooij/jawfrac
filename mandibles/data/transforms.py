from itertools import product
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from miccai.data.transforms import *


class NonNegativeCrop:

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # find smallest bounding box that encapsulates all non-negative voxels
        slices = ndimage.find_objects(intensities >= 0)[0]
        
        # crop volumes given slices
        data_dict['intensities'] = intensities[slices]
        if 'labels' in data_dict:
            data_dict['labels'] = data_dict['labels'][slices]

        # determine affine transformation from source to crop
        affine = np.eye(4)
        affine[:3, 3] -= [s.start for s in slices]
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    
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
        for key in ['intensities', 'mandible', 'labels']:
            if key not in data_dict:
                continue
            
            min_value, max_value = data_dict[key].min(), data_dict[key].max()
            data_dict[key] = ndimage.zoom(input=data_dict[key], zoom=zoom)
            data_dict[key] = np.clip(data_dict[key], min_value, max_value)

        # update voxel spacing accordingly
        data_dict['spacing'] = self.spacing

        # determine affine transformation from input to result
        affine = np.eye(4)
        affine[np.diag_indices(3)] = zoom
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
        for key in ['intensities', 'mandible', 'labels']:
            if key not in data_dict:
                continue

            shape = data_dict[key].shape
            data_dict[key] = nibabel.apply_orientation(
                arr=data_dict[key],
                ornt=orientation,
            )
        
        # update orientation to identity
        data_dict['orientation'] = nibabel.io_orientation(affine=np.eye(4))

        # determine affine transformation from input to result
        inv_affine = nibabel.orientations.inv_ornt_aff(
            ornt=orientation,
            shape=shape,
        )
        affine = np.linalg.inv(inv_affine)
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


class BonePatchIndices:

    def __init__(
        self,
        intensity_threshold: int=300,
        volume_threshold: float=0.01,
    ) -> None:
        self.intensity_thresh = intensity_threshold
        self.volume_thresh = volume_threshold
    
    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        bone_patch_idxs = []
        for patch_idxs in data_dict['patch_idxs']:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = intensities[slices]

            bone_mask = patch >= self.intensity_thresh
            if bone_mask.mean() >= self.volume_thresh:
                bone_patch_idxs.append(patch_idxs)

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = np.stack(bone_patch_idxs)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    intensity_threshold={self.intensity_thresh},',
            f'    volume_threshold={self.volume_thresh},',
            ')',
        ])


class RelativePatchCoordinates:

    def __call__(
        self,
        labels: NDArray[Any],
        patch_idxs: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # compute statistics of annotated object
        voxels = np.column_stack(labels.nonzero())
        centroid = voxels.mean(axis=0)
        scale = voxels.std(axis=0).max()

        # determine normalized relative coordinates of each patch
        centers = patch_idxs.mean(axis=2)
        patch_coords = (centers - centroid) / scale
        patch_coords[:, 0] = np.abs(patch_coords[:, 0])  # left-right symmetry

        data_dict['labels'] = labels
        data_dict['patch_idxs'] = patch_idxs
        data_dict['patch_coords'] = patch_coords

        return data_dict
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PositiveNegativeIndices:

    def __init__(
        self,
        volume_threshold: float=0.05,
    ) -> None:
        self.volume_thresh = volume_threshold

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        pos_idxs, neg_idxs = [], []
        for i, patch_idxs in enumerate(data_dict['patch_idxs']):
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # determine if patch contains sufficient foreground voxels
            if patch.mean() >= self.volume_thresh:
                pos_idxs.append(i)
            else:
                neg_idxs.append(i)

        data_dict['labels'] = labels
        data_dict['pos_idxs'] = np.array(pos_idxs)
        data_dict['neg_idxs'] = np.array(neg_idxs)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    pos_volume_threshold={self.volume_thresh},',
            ')',
        ])


class IntensityAsFeatures:

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = intensities

        intensities = intensities.clip(-1024, 3071).astype(float)

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], intensities[np.newaxis]),
            )
        else:
            data_dict['features'] = intensities[np.newaxis]

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class RandomPatchTranslate:

    def __init__(
        self,
        max_voxels: int,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max = max_voxels
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        intensities: NDArray[Any],
        patch_idxs: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # sample random translations
        trans = self.rng.integers(
            low=-self.max,
            high=self.max + 1,
            size=patch_idxs.shape[:2] + (1,),
        )

        # ensure resulting patches are contained in volume
        start_idxs, stop_idxs = (patch_idxs + trans).transpose(2, 0, 1)
        diff = (
            np.maximum(start_idxs, 0) - start_idxs
            +
            np.minimum(intensities.shape, stop_idxs) - stop_idxs
        )
        diff = diff.reshape(trans.shape)

        # apply translations to patch indices
        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = patch_idxs + trans + diff

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_voxels={self.max},',
            ')',
        ])


class RandomXAxisFlip:

    def __init__(
        self,
        p: float=0.5,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        intensities: NDArray[Any],
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # flip x-axis of volumes
        flip = 1 - 2 * (self.rng.random() < self.p)
        data_dict['intensities'] = intensities[::flip]
        data_dict['labels'] = labels[::flip]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    p={self.p},',
            ')',
        ])


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
        patch_idxs: NDArray[Any],
        pos_idxs: NDArray[np.int64],
        neg_idxs: NDArray[np.int64],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # sample at most self.max_pos_patches positive patches from scan
        pos_idxs = self.rng.permutation(pos_idxs)
        pos_idxs = pos_idxs[:self.max_pos_patches]

        # sample as many negative patches as positive patches
        neg_idxs = self.rng.permutation(neg_idxs)
        neg_idxs = neg_idxs[:pos_idxs.shape[0]]

        # select the subsample of patch indices
        pos_neg_idxs = np.concatenate((pos_idxs, neg_idxs))
        patch_idxs = patch_idxs[pos_neg_idxs]

        # crop patches from volumes
        feature_patches, label_patches = [], []
        for patch_idxs in patch_idxs:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            label_patch = labels[slices]
            label_patches.append(label_patch)

            slices = (slice(None),) + slices
            feature_patch = features[slices]
            feature_patches.append(feature_patch)

        # stack patches to make batch
        data_dict['features'] = np.stack(feature_patches)
        data_dict['labels'] = np.stack(label_patches)
        data_dict['patch_idxs'] = patch_idxs
        if 'patch_coords' in data_dict:
            data_dict['coords'] = data_dict['patch_coords'][pos_neg_idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_patches={2 * self.max_pos_patches},',
            ')',
        ])
