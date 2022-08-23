from itertools import product
from typing import Any, Callable, Dict, List

import nibabel.orientations as ornts
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
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
                data_dict[k] = torch.tensor(v.copy(), dtype=torch.bool)
            elif dtype in self.int_dtypes:
                data_dict[k] = torch.tensor(v.copy(), dtype=torch.int64)            
            elif dtype in self.float_dtypes:
                data_dict[k] = torch.tensor(v.copy(), dtype=torch.float32)
            else:
                raise ValueError(
                    'Expected a scalar or list or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Clip:

    "Clip the values of the input around the provided level."

    def __init__(
        self,
        level: float,
        width: float,
    ) -> None:
        self.low = level - width / 2
        self.high = level + width / 2

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = np.clip(intensities, self.low, self.high)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    low={self.low},',
            f'    high={self.high},',
            ')',
        ])


class IntervalNormalize:
    
    def __init__(
        self,
        low: float,
        high: float,
    ) -> None:
        self.low = low
        self.range = high - low

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        intensities = intensities - intensities.min()
        intensities = intensities / intensities.max()

        data_dict['intensities'] = (intensities * self.range) + self.low

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    low={self.low},',
            f'    high={self.low + self.range},',
            ')',
        ])

    
class RegularScale:

    def __init__(
        self,
        scale: float,
    ) -> None:
        self.scale = scale

    def __call__(
        self,
        intensities: NDArray[Any],
        zooms: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        intensities = ndimage.zoom(intensities, self.scale * zooms)

        data_dict['intensities'] = intensities
        data_dict['zooms'] = zooms

        if 'labels' in data_dict:
            labels = ndimage.zoom(data_dict['labels'], self.scale * zooms)
            data_dict['labels'] = labels

        return data_dict


class NaturalHeadPositionOrient:

    def __call__(
        self,
        intensities: NDArray[Any],
        affine: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        orientation = ornts.io_orientation(affine)
        intensities = ornts.apply_orientation(intensities, orientation)

        data_dict['intensities'] = intensities
        data_dict['affine'] = affine

        if 'labels' in data_dict:
            labels = ornts.apply_orientation(data_dict['labels'], orientation)
            data_dict['labels'] = labels

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

        data_dict['crop_slices'] = slice_coords
        data_dict['intensities'] = intensities

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
        intensity_thresh: float=0.4,
        volume_thresh: float=0.1
    ) -> None:
        self.intensity_thresh = intensity_thresh
        self.volume_thresh = volume_thresh

    def __call__(
        self,
        intensities: NDArray[Any],
        crop_slices: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        bone_slices = []
        for slices in crop_slices:
            crop = intensities[tuple(map(lambda s: slice(*s), slices))]
            crop_bone = crop >= self.intensity_thresh
            if crop_bone.mean() >= self.volume_thresh:
                bone_slices.append(slices)

        data_dict['bone_slices'] = np.stack(bone_slices)
        data_dict['intensities'] = intensities
        data_dict['crop_slices'] = crop_slices

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
        bone_slices: NDArray[Any],
        labels: NDArray[Any],
        bboxes: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        filled = np.zeros_like(labels)
        for bbox in bboxes:
            slices = tuple(map(lambda s: slice(*s), bbox))
            filled[slices] = 1

        ious = np.zeros(bone_slices.shape[0])
        for i, slices in enumerate(bone_slices):
            slices = tuple(map(lambda s: slice(*s), slices))
            ious[i] = np.any(labels[slices]) * filled[slices].mean()

        data_dict['bone_slices'] = bone_slices
        data_dict['labels'] = labels
        data_dict['bboxes'] = bboxes
        data_dict['ious'] = ious

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class IntensityAsFeatures:

    def __call__(
        self,
        intensities: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['intensities'] = intensities[np.newaxis]

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
