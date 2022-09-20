from pathlib import Path
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
            ) if stage == 'fit' else ()),
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_inputs(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        return {
            'intensities': intensities,
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_target(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.int16]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj)

        return {
            'labels': (labels == 2).astype(np.int16),
        }
