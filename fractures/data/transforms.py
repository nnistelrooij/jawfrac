from typing import Any, Dict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from sklearn.cluster import DBSCAN

from mandibles.data.transforms import *


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
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # set all intensities outside mandible to -1000
        mandible = ndimage.binary_dilation(
            input=mandible,
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=10,
        )
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
        data_dict['mandible'] = mandible[padded_slices]
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
            ')',
        ])


class ExpandLabel:

    def __init__(
        self,
        bone_iters: int,
        all_iters: int,
        bone_intensity_threshold: int=300,
    ) -> None:
        self.bone_iters = bone_iters
        self.bone_intensity_thresh = bone_intensity_threshold
        self.all_iters = all_iters

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
            out += labels.astype(np.int16)

        if self.all_iters:
            labels = ndimage.binary_dilation(
                input=labels,
                structure=ndimage.generate_binary_structure(3, 2),
                iterations=self.all_iters,
            )
            out += labels.astype(np.int16)

        data_dict['intensities'] = intensities
        data_dict['labels'] = out

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    bone_iters={self.bone_iters},',
            f'    all_iters={self.all_iters},',
            f'    bone_intensity_threshold={self.bone_intensity_thresh},',
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
