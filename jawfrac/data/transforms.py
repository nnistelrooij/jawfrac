import copy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import nibabel
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pywt
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


class MandibleCrop:

    def __init__(
        self,
        padding: Union[float, ArrayLike],
        extend: bool,
    ) -> None:
        self.padding = padding
        self.extend = extend

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # find smallest bounding box that encapsulates mandible
        slices = ndimage.find_objects(data_dict['mandible'])[0]

        # determine slices of bounding box after padding
        padded_slices = ()
        crop_shape, crop_slices = (), ()
        padding = np.ceil(self.padding / data_dict['spacing']).astype(int)
        for dim, s, pad in zip(data_dict['mandible'].shape, slices, padding):
            padded_slice = slice(max(s.start - pad, 0), min(dim, s.stop + pad))
            padded_slices += (padded_slice,)

            dim = padded_slice.stop - padded_slice.start
            if not self.extend:
                crop_shape += (dim,)
                crop_slices += (slice(None),)
                continue

            crop_shape += (s.stop - s.start + 2 * pad,)
            crop_slice = slice(
                max(0, pad - s.start),
                min(dim + max(0, pad - s.start), s.stop - s.start + 2 * pad),
            )
            crop_slices += (crop_slice,)

        # crop volumes given padded slices
        for key in ['intensities', 'mandible', 'labels']:
            if key not in data_dict:
                continue
            
            volume = np.full(crop_shape, data_dict[key].min())
            volume[crop_slices] = data_dict[key][padded_slices]
            data_dict[key] = volume

        # determine affine transformation from source to extended crop
        affine = np.eye(4)
        affine[:3, 3] -= [s.start for s in padded_slices]
        affine[:3, 3] -= [min(0, s.start - p) for s, p in zip(slices, padding)]
        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    padding={self.padding},',
            f'    extend={self.extend},',
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
        intensities: NDArray[Any],
        spacing: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # compute how much bigger results should be
        zoom = spacing / self.spacing
        
        # immediately return if zoom is not necessary
        if np.allclose(zoom, 1):
            data_dict['intensities'] = intensities
            data_dict['spacing'] = spacing

            return data_dict

        # interpolate intensities volume to given voxel spacing
        min_value, max_value = intensities.min(), intensities.max()
        intensities = ndimage.zoom(input=intensities, zoom=zoom)
        data_dict['intensities'] = intensities.clip(min_value, max_value)
        
        # interpolate mandible segmentation to given voxel spacing
        if 'mandible' in data_dict:
            data_dict['mandible'] = ndimage.zoom(
                input=data_dict['mandible'], zoom=zoom, output=float,
            ).round().astype(bool)

        # interpolate labels volume to given voxel spacing
        if 'labels' in data_dict:
            labels = np.zeros_like(intensities, dtype=data_dict['labels'].dtype)
            for label in np.unique(data_dict['labels'])[1:]:
                label_mask = ndimage.zoom(
                    input=data_dict['labels'] == label, zoom=zoom, output=float,
                ).round().astype(bool)
                labels[label_mask] = label
            
            data_dict['labels'] = labels

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
        subgrid_shape = ()
        for dim_size in intensities.shape:
            dim_idxs = list(range(0, dim_size - self.patch_size, self.stride))
            dim_idxs = np.array(dim_idxs + [dim_size - self.patch_size])

            start_idxs.append(dim_idxs)
            subgrid_shape += (dim_idxs.shape[0],)

        # determine start and stop indices of patches in all dimensions
        start_idxs = np.stack(np.meshgrid(*start_idxs, indexing='ij'), axis=3)
        stop_idxs = start_idxs + self.patch_size
        patch_idxs = np.stack((start_idxs, stop_idxs), axis=4)

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = patch_idxs
        data_dict['patch_classes'] = np.full(patch_idxs.shape[:3], -1)

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
        patches_mask = np.zeros(data_dict['patch_idxs'].shape[:3], dtype=bool)
        for i, patch_idxs_x in enumerate(data_dict['patch_idxs']):
            for j, patch_idxs_y in enumerate(patch_idxs_x):
                for k, patch_idxs_z in enumerate(patch_idxs_y):
                    slices = tuple(slice(lo, hi) for lo, hi in patch_idxs_z)
                    patch = intensities[slices]

                    bone_mask = patch >= self.intensity_thresh
                    bone_patch = bone_mask.mean() >= self.volume_thresh
                    patches_mask[i, j, k] = bone_patch

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = data_dict['patch_idxs'][patches_mask]
        data_dict['patch_classes'] = data_dict['patch_classes'][patches_mask]

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

        patch_idxs = data_dict['patch_idxs'].reshape(-1, 3, 2)
        patches_mask = np.zeros(patch_idxs.shape[0], dtype=bool)
        for i, patch_idxs in enumerate(patch_idxs):
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = intensities[slices]

            mandible_mask = mandible[slices] & (patch >= self.intensity_thresh)
            patches_mask[i] = mandible_mask.mean() >= self.volume_thresh

        data_dict['intensities'] = intensities
        data_dict['patch_idxs'] = patch_idxs[patches_mask]
        data_dict['patch_classes'] = data_dict['patch_classes'][patches_mask]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    intensity_threshold={self.intensity_thresh},',
            f'    volume_threshold={self.volume_thresh},',
            ')',
        ])


class PositiveNegativeIndices:

    def __init__(
        self,
        volume_threshold: float=0.01,
    ) -> None:
        self.volume_thresh = volume_threshold

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        patch_idxs = data_dict['patch_idxs'].reshape(-1, 3, 2)
        volumes = []
        for patch_idxs in patch_idxs:
            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # determine if patch contains sufficient foreground voxels
            volumes.append(patch.mean())

        volumes = np.array(volumes)
        patch_classes = volumes >= self.volume_thresh

        data_dict['labels'] = labels
        data_dict['patch_classes'] = patch_classes

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    pos_volume_threshold={self.volume_thresh},',
            ')',
        ])


class MandibleStatistics:

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # compute statistics of annotated object
        voxels = np.column_stack(labels.nonzero())
        centroid = voxels.mean(axis=0)
        scale = voxels.std(axis=0).max()

        data_dict['labels'] = labels
        data_dict['centroid'] = centroid
        data_dict['scale'] = scale

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class LinearFracturePatchIndices:

    def __init__(
        self,
        patch_size: int,
        fg_label: int=1,
    ) -> None:
        self.half_length = patch_size // 2
        self.fg_label = fg_label

    def bounding_boxes(
        self,
        cluster_idxs: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # determine voxel indices of centroid
        centroids = np.zeros((cluster_idxs.max(), 3), dtype=int)
        for i in range(cluster_idxs.max()):
            # compute centroid of annotated object
            voxel_idxs = np.column_stack((cluster_idxs == i + 1).nonzero())
            centroids[i] = voxel_idxs.mean(axis=0).round().astype(int)

        # determine start and stop indices of patch around centroid
        start_idxs = centroids - self.half_length
        stop_idxs = centroids + self.half_length
        patch_idxs = np.dstack((start_idxs, stop_idxs))

        # ensure resulting patches are contained in volume
        diff = (
            np.maximum(start_idxs, 0) - start_idxs
            +
            np.minimum(data_dict['labels'].shape, stop_idxs) - stop_idxs
        )
        patch_idxs += diff[..., np.newaxis]

        # update or set new patch indices and classes
        patch_classes = np.full(patch_idxs.shape[0], self.fg_label)
        if 'patch_idxs' in data_dict:
            data_dict['patch_idxs'] = np.concatenate(
                (data_dict['patch_idxs'], patch_idxs),
            )
            data_dict['patch_classes'] = np.concatenate(
                (data_dict['patch_classes'], patch_classes),
            )
        else:
            data_dict['patch_idxs'] = patch_idxs
            data_dict['patch_classes'] = patch_classes

        return data_dict

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not np.any(data_dict['labels'] == self.fg_label):
            return data_dict

        # cluster annotations using DBSCAN
        labels = (data_dict['labels'] == self.fg_label).astype(int)
        voxel_idxs = np.column_stack(labels.nonzero())
        dbscan = DBSCAN(eps=10.0, min_samples=80)
        dbscan.fit(voxel_idxs)

        # seperate each annotation with a different label
        cluster_idxs = dbscan.labels_
        labels[tuple(voxel_idxs.T)] = cluster_idxs + 1

        return self.bounding_boxes(labels, **data_dict)

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.half_length * 2},',
            ')',
        ])


class DisplacedFracturePatchIndices(LinearFracturePatchIndices):

    def __init__(
        self,
        patch_size: int,
        fg_label: int=2,
        voxel_threshold: int=1000,
    ) -> None:
        super().__init__(patch_size, fg_label)

        self.voxel_thresh = voxel_threshold

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not np.any(data_dict['labels'] == self.fg_label):
            return data_dict

        # determine connected components
        labels = data_dict['labels'] == self.fg_label
        labels, _ = ndimage.label(
            input=labels,
            structure=ndimage.generate_binary_structure(3, 1),
        )
        _, counts = np.unique(labels, return_counts=True)

        # remove connected components smaller than self.voxel_thresh
        labels[(counts < self.voxel_thresh)[labels]] = 0
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(data_dict['labels'].shape)
        
        return self.bounding_boxes(labels, **data_dict)

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    patch_size={self.half_length * 2},',
            f'    voxel_threshold={self.voxel_thresh},',
            ')',
        ])


class ExpandLabel:

    def __init__(
        self,
        bone_iters: int,
        all_iters: int,
        negative_iters: int,
        smooth: float,
        bone_intensity_threshold: int=300,
    ) -> None:
        self.bone_iters = bone_iters
        self.bone_intensity_thresh = bone_intensity_threshold
        self.all_iters = all_iters
        self.negative_iters = negative_iters
        self.smooth = smooth

    def __call__(
        self,
        intensities: NDArray[Any],
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.bone_iters:
            dilation = ndimage.binary_dilation(
                input=labels == 1,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.bone_iters,
                mask=intensities >= self.bone_intensity_thresh,
            )
            labels[dilation] = 1

        if self.all_iters:
            dilation = ndimage.binary_dilation(
                input=labels == 1,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.all_iters,
            )
            labels[dilation] = 1

        if self.negative_iters:
            frac_mask = np.any(labels[..., None] == np.array([1, 2]), axis=-1)
            dilation = ndimage.binary_dilation(
                input=frac_mask,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.negative_iters,
            )
            labels[~frac_mask & dilation] = -1

        labels = labels.astype(np.float32)
        if self.smooth:
            blur = ndimage.gaussian_filter(
                input=(labels == 1).astype(np.float32),
                sigma=self.smooth,
            )
            clip_mask = (0 < blur) & (blur <= 1)
            labels[clip_mask] = blur[clip_mask]

        data_dict['intensities'] = intensities
        data_dict['labels'] = labels

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


class FractureCentroids:

    def __init__(
        self,
        conf_thresh: float=0.1,
        voxel_thresh: int=1000,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.voxel_thresh = voxel_thresh

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        centroids = []

        # determine centroids of linear fractures
        linear_labels, _ = ndimage.label(
            input=(self.conf_thresh <= labels) & (labels < 2),
            structure=ndimage.generate_binary_structure(3, 1),
        )

        # remove components with < 1000 voxels
        unique, inverse, counts = np.unique(
            linear_labels, return_inverse=True, return_counts=True,
        )
        small_mask = (counts < self.voxel_thresh)[inverse]
        linear_labels[small_mask.reshape(linear_labels.shape)] = 0
        
        # linear centroid is voxel centroid
        for label in unique[1:]:
            voxels = np.column_stack(np.nonzero(linear_labels == label))
            centroids.append(voxels.mean(axis=0))

        # determine connected components of displaced fractures
        displaced_labels, _ = ndimage.label(
            input=labels == 2,
            structure=ndimage.generate_binary_structure(3, 1),
        )

        # remove components with < 1000 voxels
        unique, inverse, counts = np.unique(
            displaced_labels, return_inverse=True, return_counts=True,
        )
        small_mask = (counts < self.voxel_thresh)[inverse]
        displaced_labels[small_mask.reshape(displaced_labels.shape)] = 0

        # displaced centroid is closest mandible voxel from annotation centroid
        mandible_voxels = np.column_stack(np.nonzero(labels == 3))
        for label in unique[1:]:
            frac_voxels = np.column_stack(np.nonzero(displaced_labels == label))
            frac_centroid = frac_voxels.mean(axis=0)

            if mandible_voxels.shape[0] == 0:
                centroids.append(frac_centroid)
                continue

            
            dists = np.sum((mandible_voxels - frac_centroid) ** 2, axis=1)
            mandible_centroid = mandible_voxels[dists < 50 ** 2].mean(axis=0)

            # pick closest fracture voxel
            dists = np.sum((frac_voxels - mandible_centroid) ** 2, axis=1)
            if dists.min() < 50 ** 2:
                frac_centroid = frac_voxels[np.argmin(dists)]

            centroids.append(frac_centroid)

        data_dict['labels'] = labels
        data_dict['centroids'] = np.stack(centroids).astype(float)

        return data_dict
        
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    conf_thresh={self.conf_thresh},',
            f'    voxel_thresh={self.voxel_thresh},',
            ')',
        ])


class NegativeIndices:

    def __call__(
        self,
        labels: NDArray[Any],
        patch_classes: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        for i, patch_idxs in enumerate(data_dict['patch_idxs']):
            if patch_classes[i] != -1:
                continue

            # extract a patch from the volume
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch = labels[slices]
            
            # save patch without foreground voxels
            patch_classes[i] = -1 * np.any((0 < patch) & (patch <= 2))

        data_dict['labels'] = labels
        data_dict['patch_classes'] = patch_classes

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


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
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.rng.random() >= self.p:
            return data_dict

        # flip x-axis of volumes
        for key in ['intensities', 'mandible', 'labels']:
            if key not in data_dict:
                continue
            
            shape = data_dict[key].shape
            data_dict[key] = data_dict[key][::-1]

        # move patches to other side of volume
        patch_idxs = data_dict['patch_idxs'].copy()
        patch_idxs[:, 0] = np.fliplr(shape[0] - patch_idxs[:, 0])
        data_dict['patch_idxs'] = patch_idxs

        if 'centroid' in data_dict:
            centroid = data_dict['centroid'].copy()
            centroid[0] = data_dict['labels'].shape[0] - centroid[0]
            data_dict['centroid'] = centroid

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    p={self.p},',
            ')',
        ])


class RandomGammaAdjust:

    def __init__(
        self,
        low: float=0.8,
        high: float=1.2,
        rng: Optional[np.random.Generator]=None
    ) -> None:
        self.low = low
        self.range = high - low
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        features: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        gamma = self.low + self.range * self.rng.random()

        features = (features + 1024) / 4120
        features = features ** gamma
        features = (features * 4120) - 1024

        data_dict['features'] = features

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '()',
            f'    low={self.low},',
            f'    high={self.low + self.range / 2},',
            ')',
        ])


class RandomPatchTranslate:

    def __init__(
        self,
        max_voxels: int,
        classes: List[int]=[0, 1, 2],
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max = max_voxels
        self.classes = np.array(classes).reshape(-1, 1)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        intensities: NDArray[Any],
        patch_classes: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # determine which patches should be translated
        patch_mask = np.any(patch_classes[np.newaxis] == self.classes, axis=0)
        patch_idxs = data_dict['patch_idxs'][patch_mask]

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
        data_dict['patch_idxs'][patch_mask] = patch_idxs + trans + diff
        data_dict['patch_classes'] = patch_classes

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_voxels={self.max},',
            f'    classes={self.classes.flatten().tolist()},',
            ')',
        ])


class HaarTransform:

    def denoise(
        self,
        intensities,
    ) -> NDArray[Any]:
        coeffs_1 = pywt.dwtn(intensities, 'sym4')
        for k_1 in coeffs_1.copy():
            # do not change approximation
            if 'd' not in k_1:
                coeffs_2 = pywt.dwtn(coeffs_1[k_1], 'haar')
                coeffs_1[k_1] = pywt.idwtn(coeffs_2, 'haar')

                continue

            coeffs_2 = pywt.dwtn(coeffs_1[k_1], 'haar')
            for k_2, coeffs in coeffs_2.copy().items():
                T = np.std(coeffs) * np.sqrt(2 * np.log(coeffs.size))

                G = np.zeros_like(coeffs)
                G[coeffs >= T] = coeffs[coeffs >= T] - T
                G[coeffs <= -T] = coeffs[coeffs <= -T] + T

                coeffs_2[k_2] = G

            coeffs_1[k_1] = pywt.idwtn(coeffs_2, 'haar')

        # get ouptut with same shape as input
        out = pywt.idwtn(coeffs_1, 'sym4')
        out = out[tuple(slice(None, dim) for dim in intensities.shape)]        

        return out

    def edge_detection(
        self,
        intensities: NDArray[Any],
    ) -> NDArray[Any]:
        dx = ndimage.sobel(intensities, axis=0)
        dy = ndimage.sobel(intensities, axis=1)
        dz = ndimage.sobel(intensities, axis=2)
        mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        intensities = intensities - (mag / mag.max())

        return intensities

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # get intensities between 0 and 1
        dtype = intensities.dtype
        intensities = (intensities + 1024) / 4120

        # denoise intensities using Haar transform
        intensities = self.denoise(intensities)

        # accentuate edges
        intensities = self.edge_detection(intensities)

        # get intensities back between -1024 and 3096
        intensities = (intensities * 4120) - 1024
        intensities = intensities.clip(-1024, 3096)

        data_dict['intensities'] = intensities.astype(dtype)

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class RelativePatchCoordinates:

    def __call__(
        self,
        patch_idxs: NDArray[Any],
        centroid: NDArray[Any],
        scale: float,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # determine normalized relative coordinates of each patch
        centers = patch_idxs.mean(axis=2)
        patch_coords = (centers - centroid) / scale
        patch_coords[:, 0] = np.abs(patch_coords[:, 0])  # left-right symmetry

        data_dict['patch_idxs'] = patch_idxs
        data_dict['centroid'] = centroid
        data_dict['scale'] = scale
        data_dict['patch_coords'] = patch_coords

        return data_dict
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class IntensityAsFeatures:

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = intensities

        intensities = intensities.astype(float)
        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], intensities[np.newaxis]),
            )
        else:
            data_dict['features'] = intensities[np.newaxis]

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class PositiveNegativePatches:

    def __init__(
        self,
        max_patches: int,
        pos_classes: List[int]=[1],
        ignore_outside: bool=True,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max_pos_patches = max_patches // 2
        self.pos_classes = np.array([pos_classes]).T
        self.ignore_outside = ignore_outside
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        intensities: NDArray[np.int16],
        features: NDArray[np.float64],
        labels: NDArray[Any],
        patch_idxs: NDArray[Any],
        patch_classes: NDArray[np.int64],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # sample at most self.max_pos_patches positive patches from scan
        pos_mask = np.any(patch_classes == self.pos_classes, axis=0)
        pos_idxs = pos_mask.nonzero()[0]
        pos_idxs = self.rng.permutation(pos_idxs)
        pos_idxs = pos_idxs[:self.max_pos_patches]

        # sample as many negative patches as positive patches
        neg_idxs = (patch_classes == 0).nonzero()[0]
        neg_idxs = self.rng.permutation(neg_idxs)
        neg_idxs = neg_idxs[:pos_idxs.shape[0]]

        # select the subsample of patch indices
        pos_neg_idxs = np.concatenate((pos_idxs, neg_idxs))

        # crop patches from volumes
        patch_features, patch_masks = [], []
        for patch_idxs in patch_idxs[pos_neg_idxs]:
            slices = tuple(slice(start, stop) for start, stop in patch_idxs)
            patch_mask = labels[slices].astype(float, copy=True)
            patch_mask[(patch_mask < 0) | (1 < patch_mask)] = 0
            if self.ignore_outside:
                patch_mask[intensities[slices] == intensities.min()] = -1
            patch_masks.append(patch_mask)

            slices = (slice(None),) + slices
            patch_features.append(features[slices])

        # stack patches to make batch
        data_dict['intensities'] = intensities
        data_dict['features'] = np.stack(patch_features)
        data_dict['masks'] = np.stack(patch_masks)
        data_dict['classes'] = patch_classes[pos_neg_idxs]
        if 'patch_coords' in data_dict:
            data_dict['coords'] = data_dict['patch_coords'][pos_neg_idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_patches={2 * self.max_pos_patches},',
            f'    pos_classes={self.pos_classes.flatten().tolist()},',
            f'    ignore_outside={self.ignore_outside},',
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
                    'Expected a scalar or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
