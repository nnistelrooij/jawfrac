import copy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from sklearn.cluster import DBSCAN
import torch
from torchtyping import TensorType


class Compose:

    def __init__(
        self,
        *transforms: List[Callable[..., Dict[str, Any]]],
    ):
        self.transforms = transforms

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        for t in self.transforms:
            data_dict = t(**data_dict)
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            *[
                '    ' + repr(t).replace('\n', '\n    ') + ','
                for t in self.transforms
            ],
            ')',
        ])


class ToTensor:

    def __init__(
        self,
        bool_dtypes: List[np.dtype]=[bool, np.bool8],
        int_dtypes: List[np.dtype]=[int, np.int16, np.uint16, np.int32, np.int64],
        float_dtypes: List[np.dtype]=[float, np.float32, np.float64],
    ) -> None:
        self.bool_dtypes = bool_dtypes
        self.int_dtypes = int_dtypes
        self.float_dtypes = float_dtypes

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, TensorType[..., Any]]:
        for k, v in data_dict.items():
            dtype = v.dtype if isinstance(v, np.ndarray) else type(v)
            if dtype in self.bool_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.bool)
            elif dtype in self.int_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.int64)            
            elif dtype in self.float_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.float32)
            else:
                raise ValueError(
                    'Expected a scalar or list or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class MandibleCrop:

    def __init__(
        self,
        padding: Union[float, ArrayLike],
        manifold: bool,
    ) -> None:
        self.padding = padding
        self.manifold = manifold

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # set all intensities outside mandible to -1000
        mandible = ndimage.binary_dilation(
            input=data_dict['mandible'],
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=10,
        )
        if self.manifold:
            intensities[~mandible] = -1024

        # find smallest bounding box that encapsulates mandible
        slices = ndimage.find_objects(mandible)[0]

        # determine slices of bounding box after padding
        padded_slices = ()
        padding = np.ceil(self.padding / data_dict['spacing']).astype(int)
        for s, pad in zip(slices, padding):
            padded_slice = slice(max(s.start - pad, 0), s.stop + pad)
            padded_slices += (padded_slice,)

        # crop volumes given padded slices
        data_dict['intensities'] = intensities[padded_slices]
        data_dict['mandible'] = data_dict['mandible'][padded_slices]
        if 'labels' in data_dict:
            data_dict['labels'] = data_dict['labels'][padded_slices]

        # determine affine transformation from source to crop
        affine = np.eye(4)
        affine[:3, 3] -= [s.start for s in padded_slices]
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    padding={self.padding},',
            f'    manifold={self.manifold},',
            ')',
        ])


class ExpandLabel:

    def __init__(
        self,
        bone_iters: int,
        all_iters: int,
        smooth: bool,
        bone_intensity_threshold: int=300,
    ) -> None:
        self.bone_iters = bone_iters
        self.bone_intensity_thresh = bone_intensity_threshold
        self.all_iters = all_iters
        self.smooth = smooth

    def __call__(
        self,
        intensities: NDArray[Any],
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = labels > 0
        out = labels.astype(np.int16)

        if self.bone_iters:
            labels = ndimage.binary_dilation(
                input=labels,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.bone_iters,
                mask=intensities >= self.bone_intensity_thresh,
            )
            out = self.smooth * out + labels.astype(np.int16)

        if self.all_iters:
            labels = ndimage.binary_dilation(
                input=labels,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.all_iters,
            )
            out = self.smooth * out + labels.astype(np.int16)

        data_dict['intensities'] = intensities
        data_dict['labels'] = out

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    bone_iters={self.bone_iters},',
            f'    bone_intensity_threshold={self.bone_intensity_thresh},',
            f'    all_iters={self.all_iters},',
            f'    smooth={self.smooth},',
            ')',
        ])


class NegativeIndices:

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        neg_idxs = []
        for i, patch_idxs in enumerate(data_dict['patch_idxs']):
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # save patch without foreground voxels
            if not np.any(patch):
                neg_idxs.append(i)            

        data_dict['labels'] = labels
        data_dict['neg_idxs'] = np.array(neg_idxs)

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PositivePatchIndices:

    def __init__(
        self,
        patch_size: int,
    ) -> None:
        self.half_length = patch_size // 2

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # cluster annotations using DBSCAN
        voxel_idxs = np.column_stack(labels.nonzero())
        dbscan = DBSCAN(eps=10.0, min_samples=80)
        dbscan.fit(voxel_idxs)

        # seperate each annotation with a different label
        cluster_idxs = dbscan.labels_
        labels[tuple(voxel_idxs.T)] = cluster_idxs + 1

        pos_patch_idxs = []
        for i in range(1, labels.max() + 1):
            # compute centroid of annotated object
            voxel_idxs = np.column_stack((labels == i).nonzero())
            centroid = voxel_idxs.mean(axis=0).round().astype(int)

            # determine start and stop indices of patch around centroid
            start = centroid - self.half_length
            stop = centroid + self.half_length
            patch_idxs = np.column_stack((start, stop))

            pos_patch_idxs.append(patch_idxs)

        # ensure resulting patches are contained in volume
        pos_patch_idxs = np.stack(pos_patch_idxs)
        start_idxs, stop_idxs = pos_patch_idxs.transpose(2, 0, 1)
        diff = (
            np.maximum(start_idxs, 0) - start_idxs
            +
            np.minimum(labels.shape, stop_idxs) - stop_idxs
        )
        pos_patch_idxs += diff[..., np.newaxis]

        data_dict['labels'] = labels
        data_dict['pos_idxs'] = np.arange(pos_patch_idxs.shape[0])
        if 'patch_idxs' in data_dict:
            data_dict['patch_idxs'] = np.concatenate(
                (pos_patch_idxs, data_dict['patch_idxs']),
            )
        else:
            data_dict['patch_idxs'] = pos_patch_idxs

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.half_length * 2},',
            ')',
        ])


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
            
            volume = data_dict[key]
            volume = ndimage.zoom(input=volume, zoom=zoom, output=float)
            volume = volume.clip(data_dict[key].min(), data_dict[key].max())
            data_dict[key] = volume.round().astype(data_dict[key].dtype)

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


class MandiblePatchIndices:

    def __init__(
        self,
        intensity_threshold: int=300,
        volume_threshold: float=0.01,
    ) -> None:
        self.intensity_thresh = intensity_threshold
        self.volume_thresh = volume_threshold

    def __call__(
        self,
        intensities,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        mandible = ndimage.binary_dilation(
            input=data_dict['mandible'],
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=10,
        )

        mandible_patch_idxs = []
        for patch_idxs in data_dict['patch_idxs']:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = intensities[slices]

            mandible_mask = mandible[slices] & (patch >= self.intensity_thresh)
            if mandible_mask.mean() >= self.volume_thresh:
                mandible_patch_idxs.append(patch_idxs)

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = np.stack(mandible_patch_idxs)

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
