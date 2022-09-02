from itertools import product
from typing import Any, Dict

import nibabel.orientations as ornts
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from miccai.data.transforms import *


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
        affine: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # set all intensities outside mandible to -1000
        mask = ndimage.binary_dilation(
            input=mandible,
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=10,
        )
        intensities[~mask] = -1000

        # crop volume to largest annotated connected component
        labels, num_labels = ndimage.label(mandible)
        sizes = ndimage.sum(mandible, labels, range(1, num_labels + 1))
        bbox = ndimage.find_objects(labels == (sizes.argmax() + 1))[0]

        spacing = np.abs(affine[:, :3].sum(axis=0))
        padding = np.ceil(self.padding / spacing).astype(int)

        slices = ()
        for slice_, pad in zip(bbox, padding):
            start = max(slice_.start - pad, 0)
            end = slice_.stop + pad
            slices += (slice(start, end),)

        data_dict['intensities'] = intensities[slices]
        data_dict['mandible'] = mandible[slices]
        data_dict['affine'] = affine

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
        self.spacing = spacing

    def __call__(
        self,
        intensities: NDArray[Any],
        mandible: NDArray[Any],
        affine: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        zooms = np.abs(np.sum(affine[:, :3], axis=0))
        intensities = ndimage.zoom(intensities, zooms / self.spacing)
        mandible = ndimage.zoom(mandible, zooms / self.spacing)

        data_dict['intensities'] = intensities
        data_dict['mandible'] = mandible
        data_dict['affine'] = affine

        if 'labels' in data_dict:
            labels = ndimage.zoom(data_dict['labels'], zooms / self.spacing)
            data_dict['labels'] = labels

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
        intensities: NDArray[Any],
        mandible: NDArray[Any],
        affine: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        orientation = ornts.io_orientation(affine)
        intensities = ornts.apply_orientation(intensities, orientation)
        mandible = ornts.apply_orientation(mandible, orientation)

        data_dict['intensities'] = intensities
        data_dict['affine'] = affine

        if 'labels' in data_dict:
            labels = ornts.apply_orientation(data_dict['labels'], orientation)
            data_dict['labels'] = labels

        import nibabel
        img = nibabel.Nifti1Image(intensities, np.eye(4))
        nibabel.save(img, '/home/mka3dlab/Documents/fractures/test.nii.gz')
        img = nibabel.Nifti1Image(labels, np.eye(4))
        nibabel.save(img, '/home/mka3dlab/Documents/fractures/frac.nii.gz')

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class CropCenters:

    def __init__(
        self,
        crop_size: int,
        stride: int,
    ) -> None:
        self.crop_size = crop_size
        self.stride = stride

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        xyz_coords = []
        for dim in intensities.shape:
            start_coords = np.arange(0, dim - self.crop_size // 2, self.stride)
            start_coords = np.clip(start_coords, 0, dim - self.crop_size)

            xyz_coords.append(start_coords)

        crop_coords = np.stack(list(product(*xyz_coords)))
        slice_coords = np.dstack((crop_coords, crop_coords + self.crop_size))

        data_dict['intensities'] = intensities
        data_dict['slices'] = slice_coords

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    half_length={self.half_length},',
            ')',
        ])


class BoneCenters:

    def __init__(
        self,
        intensity_thresh: float=500,
        volume_thresh: float=0.1
    ) -> None:
        self.intensity_thresh = intensity_thresh
        self.volume_thresh = volume_thresh

    def __call__(
        self,
        intensities: NDArray[Any],
        slices: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        bone_slices = []
        for slices in slices:
            crop = intensities[tuple(map(lambda s: slice(*s), slices))]
            crop_bone = crop >= self.intensity_thresh
            if crop_bone.mean() >= self.volume_thresh:
                bone_slices.append(slices)

        data_dict['intensities'] = intensities
        data_dict['slices'] = np.stack(bone_slices)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    intensity_thresh={self.intensity_thresh},',
            f'    volume_thresh={self.volume_thresh},',
            ')',
        ])


class BoundingBoxes:

    def __call__(
        self,
        labels: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        bboxes = []
        for obj in ndimage.find_objects(labels):
            bboxes.append(np.array([
                [obj[0].start, obj[0].stop],
                [obj[1].start, obj[1].stop],
                [obj[2].start, obj[2].stop],
            ]))

        data_dict['labels'] = labels
        data_dict['bboxes'] = np.stack(bboxes)

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class IntersectionOverUnion:

    def __call__(
        self,
        labels: NDArray[Any],
        slices: NDArray[Any],
        bboxes: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        filled = np.zeros_like(labels)
        for bbox in bboxes:
            crop = tuple(map(lambda s: slice(*s), bbox))
            filled[crop] = 1

        ious = np.zeros(slices.shape[0])
        for i, crop in enumerate(slices):
            crop = tuple(map(lambda s: slice(*s), crop))
            ious[i] = np.any(labels[crop]) * filled[crop].mean()

        data_dict['labels'] = labels
        data_dict['slices'] = slices
        data_dict['bboxes'] = bboxes
        data_dict['ious'] = ious

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
        intensities -= self.min
        intensities /= self.max - self.min
        intensities = (intensities * self.range) + self.low

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
