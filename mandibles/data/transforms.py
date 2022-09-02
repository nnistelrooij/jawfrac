from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial.transform import Rotation

from miccai.data.transforms import *


class ToPointCloud:

    def __init__(
        self,
        intensity_thresh: float=500,
        component_thresh: int=1000,
    ) -> None:
        self.intensity_thresh = intensity_thresh
        self.component_thresh = component_thresh

    def __call__(
        self,
        intensities: NDArray[Any],
        affine: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # only keep voxels with sufficient HU intensity
        mask = intensities >= self.intensity_thresh

        # remove small connected components
        label_idxs, _ = ndimage.label(mask)
        _, inverse, counts = np.unique(
            label_idxs, return_inverse=True, return_counts=True,
        )

        small_components = counts < self.component_thresh
        mask[small_components[inverse.reshape(mask.shape)]] = False

        if 'labels' in data_dict:
            labels = data_dict['labels']

            # dilate label to compensate for under-segmentations
            for idx in range(1, labels.max() + 1):
                binary_labels = ndimage.binary_dilation(
                    input=labels == idx,
                    structure=ndimage.generate_binary_structure(3, 3),
                    iterations=2,
                    mask=mask & (labels == 0),
                )
                labels[binary_labels] = idx

            # remove small connected components from background
            label_idxs, _ = ndimage.label(mask & (labels == 0))
            _, inverse, counts = np.unique(
                label_idxs, return_inverse=True, return_counts=True,
            )

            small_components = counts < self.component_thresh
            mask[small_components[inverse.reshape(mask.shape)]] = False

            # save labels of remaining voxels as COO features
            data_dict['labels'] = labels[mask].flatten()        
        
        # compute subject-centered coordinates of remaining voxels
        coords = np.column_stack(mask.nonzero())
        coords = np.einsum('kj,ij->ki', coords, affine[:3, :3])

        affine[:3, 3] = 0

        data_dict['intensities'] = intensities[mask].flatten()
        data_dict['affine'] = affine
        data_dict['points'] = coords
        data_dict['point_count'] = coords.shape[0]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    intensity_thresh={self.intensity_thresh},',
            f'    component_thresh={self.component_thresh},',
            ')',
        ])


class Rotate:

    def __init__(
        self,
        axes: str,
        degrees: List[float],
    ) -> None:
        rot = np.eye(4)
        rot[:-1, :-1] = Rotation.from_euler(
            seq=axes, angles=degrees, degrees=True,
        ).as_matrix()

        self.axes = axes
        self.degrees = degrees
        self.rot_matrix = rot

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        points = np.einsum('kj,ij->ki', points, self.rot_matrix[:3, :3])
        affine = self.rot_matrix @ data_dict.get('affine', np.eye(4))
        
        data_dict['points'] = points
        data_dict['affine'] = affine

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    axes={self.axes},',
            f'    degrees={self.degrees},',
            ')'
        ])


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

        if 'features' in data_dict:
            data_dict['features'] = np.column_stack(
                (data_dict['features'], intensities),
            )
        else:
            data_dict['features'] = intensities[:, np.newaxis]

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


class NearestNeighborCrop:

    def __init__(
        self,
        neigbors: int,
        seed_fn=lambda points: np.random.randint(points.shape[0]),
    ) -> None:
        self.neigbors = neigbors
        self.seed_fn = seed_fn

    def __call__(
        self,
        points: NDArray[Any],
        point_count: int,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if point_count <= self.neigbors:
            data_dict['points'] = points
            data_dict['point_count'] = point_count

            return data_dict

        seed = self.seed_fn(points)
        rel_coords = points - points[seed]
        sq_dists = np.sum(rel_coords ** 2, axis=-1)
        crop_idxs = np.argsort(sq_dists)[:self.neigbors]

        data_dict['points'] = points[crop_idxs]
        data_dict['point_count'] = crop_idxs.shape[0]
        
        for key in ['features', 'labels', 'instances']:
            if key not in data_dict:
                continue

            data_dict[key] = data_dict[key][crop_idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    neighbors={self.neigbors},',
            ')',
        ])
        

class RandomTranslate:

    def __init__(
        self,
        max_dist: Union[float, ArrayLike],
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max = max_dist
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        trans = self.rng.random(size=3)
        trans = (trans * 2 * self.max) - self.max

        data_dict['points'] = points + trans

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max={self.max},',
            ')',
        ])
