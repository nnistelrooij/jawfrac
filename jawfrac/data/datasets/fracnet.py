from pathlib import Path
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import NDArray

from jawfrac.data.datasets.base import VolumeDataset
import jawfrac.data.transforms as T


class FracNetDataset(VolumeDataset):
    """Dataset to load head scans with fracture segmentations."""

    def __init__(
        self,
        stage: str,
        regular_spacing: float,
        patch_size: int,
        expand_label: Dict[str, int],
        stride: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.NonNegativeCrop(),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            T.BonePatchIndices(),
            *((
                T.LinearFracturePatchIndices(patch_size=patch_size),
                T.DisplacedFracturePatchIndices(patch_size=patch_size),
                T.ExpandLabel(**expand_label),
                T.NegativeIndices(),
            ) if stage == 'fit' else ()),
            T.ExpandLabel(**expand_label) if stage == 'test' else dict,
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_inputs(
        self,
        file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / file)
        intensities = np.asarray(img.dataobj)

        # convert 8-bit to 12-bit
        if intensities.min() == 0 and intensities.max() <= 255:
            center = intensities[intensities > 0].mean()
            intensities = (intensities - center) / 255 * 4095

        return {
            'intensities': intensities.astype(np.int16),
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_target(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.bool8]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj)

        return {
            'labels': labels.astype(np.int16),
        }
