from itertools import product
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from miccai.data.transforms import *


class ExpandLabel:

    def __init__(
        self,
        fg_iters: int,
        fg_intensity_thresh: float,
        all_iters: int,
    ) -> None:
        self.fg_iters = fg_iters
        self.fg_intensity_thresh = fg_intensity_thresh
        self.all_iters = all_iters

    def __call__(
        self,
        intensities: NDArray[Any],
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = labels > 0

        if self.fg_iters:
            labels = ndimage.binary_dilation(
                input=labels,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.fg_iters,
                mask=intensities >= self.fg_intensity_thresh,
            )

        if self.all_iters:
            labels = ndimage.binary_dilation(
                input=labels,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.all_iters,
            )

        data_dict['intensities'] = intensities
        data_dict['labels'] = labels.astype(int)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    bone_iters={self.fg_iters},',
            f'    all_iters={self.all_iters},',
            ')',
        ])


class MandibleCrop:

    def __init__(
        self,
        padding: Union[float, ArrayLike],
    ) -> None:
        self.padding = padding

    def __call__(
        self,
        intensities: NDArray[Any],
        mandible: NDArray[Any],
        spacing: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # set all intensities outside mandible to -1000
        mask = ndimage.binary_dilation(
            input=mandible,
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=10,
        )
        intensities[~mask] = -1000

        # find bounding box of largest annotated connected component
        labels, num_labels = ndimage.label(mandible)
        sizes = ndimage.sum_labels(mandible, labels, range(1, num_labels + 1))
        bbox = ndimage.find_objects(labels == (sizes.argmax() + 1))[0]

        # determine slices of bounding box after padding
        slices = ()
        padding = np.ceil(self.padding / spacing).astype(int)
        for s, pad in zip(bbox, padding):
            slices += (slice(max(s.start - pad, 0), s.stop + pad),)

        # determine affine transformation from source to crop
        affine = np.eye(4)
        affine[:3, 3] -= [s.start for s in slices]

        data_dict['intensities'] = intensities[slices]
        data_dict['mandible'] = mandible[slices]
        data_dict['spacing'] = spacing
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        if 'labels' in data_dict:
            data_dict['labels'] = data_dict['labels'][slices]

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
        for key in ['intensities', 'mandible', 'labels']:
            if key not in data_dict:
                continue

            data_dict[key] = ndimage.zoom(input=data_dict[key], zoom=zoom)

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
        for key in ['intensities', 'mandible', 'labels']:
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


class BonePatchIndices:

    def __init__(
        self,
        intensity_thresh: float,
        volume_thresh: float,
    ) -> None:
        self.intensity_thresh = intensity_thresh
        self.volume_thresh = volume_thresh

    def __call__(
        self,
        intensities: NDArray[Any],
        patch_idxs: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        bone_patch_idxs = []
        for patch_idxs in patch_idxs:
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = intensities[slices]

            # determine if there is enough bone
            patch_bone = patch >= self.intensity_thresh
            if patch_bone.mean() < self.volume_thresh:
                continue

            # save patch with bone
            bone_patch_idxs.append(patch_idxs)

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = np.stack(bone_patch_idxs)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    intensity_thresh={self.intensity_thresh},',
            f'    volume_thresh={self.volume_thresh},',
            ')',
        ])


class NegativePatchIndices:

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        neg_patch_idxs = []
        for patch_idxs in data_dict['patch_idxs']:
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # determine if patch contains foreground voxels
            if np.any(patch):
                continue

            # save patch without foreground voxels
            neg_patch_idxs.append(patch_idxs)

        data_dict['neg_patch_idxs'] = np.stack(neg_patch_idxs)
        data_dict['labels'] = labels

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ForegroundPatchIndices:

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
        fg_patch_idxs = []
        for i in range(1, labels.max() + 1):
            # compute centroid of foreground object
            voxel_idxs = np.column_stack((labels == i).nonzero())
            centroid = voxel_idxs.mean(axis=0).round().astype(int)

            # determine start and stop indices of patch around centroid
            start = np.maximum(centroid - self.half_length, 0)
            stop = np.minimum(labels.shape, centroid + self.half_length)
            patch_idxs = np.column_stack((start, stop))

            fg_patch_idxs.append(patch_idxs)

        data_dict['labels'] = labels
        data_dict['fg_patch_idxs'] = np.stack(fg_patch_idxs)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.half_length * 2},',
            ')',
        ])


class PositivePatchIndices:

    def __init__(
        self,
        patch_size: int,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.patch_size = patch_size
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        fg_patch_idxs: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        pos_patch_idxs = []
        for patch_idxs in fg_patch_idxs:
            # sample patch  inside foreground patch
            start_idxs = self.rng.integers(
                low=patch_idxs[:, 0],
                high=patch_idxs[:, 1] - self.patch_size,
            )
            stop_idxs = start_idxs + self.patch_size
            patch_idxs = np.column_stack((start_idxs, stop_idxs))
            
            pos_patch_idxs.append(patch_idxs)

        data_dict['fg_patch_idxs'] = fg_patch_idxs
        data_dict['pos_patch_idxs'] = np.stack(pos_patch_idxs)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + ')',
            f'    patch_size={self.patch_size},',
            ')',
        ])


class IntersectionOverUnion:

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        purities, integrities = np.zeros((2, patch_idxs.shape[0]))
        for i, patch_idxs in enumerate(data_dict['patch_idxs']):
            # determine if there is any object in patch
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            if not np.any(labels[slices]):
                continue
            
            # compute nr. voxels comprising object, patch, and object in patch
            obj_idx = labels[slices].max()
            obj_numel = (labels == obj_idx).sum()
            patch_numel = np.prod(patch_idxs[:, 1] - patch_idxs[:, 0])
            patch_obj_numel = (labels[slices] == obj_idx).sum()

            # purity =/= precision, integrity =/= recall
            purities[i] = patch_obj_numel / patch_numel
            integrities[i] = patch_obj_numel / obj_numel
        
        data_dict['purities'] = purities
        data_dict['integrities'] = integrities
        data_dict['ious'] = 1 / (1 / purities + 1 / integrities - 1)

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class IntensityAsFeatures:

    def __init__(
        self,
        level: float=450.0,
        width: float=1100.0,
        low: float=-1.0,
        high: float=1.0,
    ) -> None:
        self.min = level - width / 2
        self.max = level + width / 2
        self.low = low
        self.range = high - low

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = intensities

        intensities = intensities.clip(self.min, self.max)
        intensities = (intensities - self.min) / (self.max - self.min)
        intensities = (intensities * self.range) + self.low

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], intensities),
            )
        else:
            data_dict['features'] = intensities[np.newaxis]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    level={(self.min + self.max) / 2},',
            f'    width={self.max - self.min},',
            f'    low={self.low},',
            f'    high={self.low + self.range},',
            ')',
        ])


class RelativeXYZAsFeatures:

    def __call__(
        self,
        mandible: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        voxels = np.column_stack(mandible.nonzero())
        centroid = voxels.mean(axis=0)

        XYZ = np.meshgrid(*map(range, mandible.shape), indexing='ij')
        XYZ = np.stack(XYZ, axis=-1)
        
        rel_XYZ = XYZ - centroid
        rel_XYZ = rel_XYZ.transpose(3, 0, 1, 2)

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], rel_XYZ),
            )
        else:
            data_dict['features'] = rel_XYZ

        data_dict['mandible'] = mandible

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
