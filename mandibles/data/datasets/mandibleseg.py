from pathlib import Path
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import NDArray

from miccai.data.datasets.base import MeshDataset
import mandibles.data.transforms as T


class MandibleSegDataset(MeshDataset):
    """Dataset to load mandibular CT scans with fracture segmentations."""

    # MEAN = [55.3963, -34.5440, -16.5498]
    # STD = 20.1196
    MEAN = [-62.88073436 -62.03004933 -56.87611781]
    STD = 34.18731546

    def __init__(
        self,
        foreground_hu_threshold: float,
        downsample_voxel_size: float,
        downsample_max_points: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.ToPointCloud(intensity_thresh=foreground_hu_threshold),
            T.Rotate('xy', [180, 90]),  # Fabian
            # Rotate('yx', [180, 90]),  # Sophie
            T.ZScoreNormalize(self.MEAN, self.STD),
            T.IntensityAsFeatures(),
            T.XYZAsFeatures(),
            T.PointCloudDownsample(
                voxel_size=downsample_voxel_size,
                inplace=True,
            ),
            T.NearestNeighborCrop(
                neigbors=downsample_max_points,
                seed_fn=lambda points: points[:, 2].argmax(),
            ),
        )

        super().__init__(pre_transform=pre_transform, **kwargs)

    def load_scan(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        return {
            'intensities': intensities,
            'affine': img.affine,
            'shape': np.array(img.shape),
        }

    def load_annotation(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.int64]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj, dtype=np.int16)

        return {
            'labels': labels,
        }
