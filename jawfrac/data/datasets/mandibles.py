from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import NDArray

from jawfrac.data.datasets.base import VolumeDataset
import jawfrac.data.transforms as T


class MandibleSegDataset(VolumeDataset):
    """Dataset to load head scans with mandible segmentations."""

    def __init__(
        self,
        stage: str,
        regular_spacing: float,
        patch_size: int,
        stride: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.NonNegativeCrop(),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            *((
                T.BonePatchIndices(),
                T.PositiveNegativeIndices(),
                T.MandibleStatistics(),
            ) if stage == 'fit' else ()),
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_inputs(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        counter = perf_counter()

        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        # convert 8-bit to 12-bit
        if intensities.min() == 0 and intensities.max() == 255:
            center = intensities[intensities > 0].mean()
            intensities = (intensities - center) / 255 * 4095

        # clip intensities to sensible range
        intensities = intensities.clip(-1024, 3096)

        return {
            'intensities': intensities.astype(np.int16),
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
            'counter': counter,
        }

    def load_target(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.bool8]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj)

        return {
            'labels': labels == 2,
        }
