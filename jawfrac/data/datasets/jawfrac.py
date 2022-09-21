from pathlib import Path
from typing import Any, Dict, List

import nibabel
import numpy as np
from numpy.typing import NDArray

from jawfrac.data.datasets.base import VolumeDataset
import jawfrac.data.transforms as T


class JawFracDataset(VolumeDataset):
    """Dataset to load mandible scans with fracture segmentations."""

    def __init__(
        self,
        stage: str,
        mandible_crop_padding: float,
        regular_spacing: float,
        patch_size: int,
        stride: int,
        expand_label: Dict[str, int],
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.MandibleCrop(padding=mandible_crop_padding),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            T.BonePatchIndices(),
            *((
                T.PositivePatchIndices(patch_size=patch_size),
                T.NegativeIndices(),
            ) if stage == 'fit' else ()),
            T.ExpandLabel(**expand_label) if stage != 'predict' else dict,
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_inputs(
        self,
        scan_file: Path,
        mandible_file: Path,
    ) -> Dict[str, NDArray[Any]]:
        img = nibabel.load(self.root / scan_file)
        intensities = np.asarray(img.dataobj)

        # convert 8-bit to 12-bit
        if intensities.min() == 0 and intensities.max() == 255:
            intensities = intensities / 255 * 4095 - 1024

        seg = nibabel.load(self.root / mandible_file)
        mask = np.asarray(seg.dataobj) == 1

        print(scan_file.parent.stem)

        return {
            'intensities': intensities.astype(np.int16),
            'mandible': mask,
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
            'labels': labels.astype(np.int16),
        }
