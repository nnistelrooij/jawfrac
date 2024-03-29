from pathlib import Path
from typing import Any, Dict, Union

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
        mandible_crop: Dict[str, Union[bool, float]],
        regular_spacing: float,
        haar_transform: bool,
        patch_size: int,
        stride: int,
        expand_label: Dict[str, int],
        pass_affine: bool,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.MandibleCrop(**mandible_crop),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            *((T.HaarTransform(),) if haar_transform else ()),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            *((
                T.BonePatchIndices(),
                T.LinearFracturePatchIndices(patch_size=patch_size),
                T.DisplacedFracturePatchIndices(patch_size=patch_size),
                T.ExpandLabel(**expand_label),
                T.NegativeIndices(),
            ) if stage == 'fit' else ()),
            T.ExpandLabel(**expand_label) if stage == 'test' else dict,
        )

        self.spacing = (regular_spacing,)*3
        self.pass_affine = pass_affine

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
            center = intensities[intensities > 0].mean()
            intensities = (intensities - center) / 255 * 4095

        # clip intensities to sensible range
        intensities = intensities.clip(-1024, 3096)

        seg = nibabel.load(self.root / mandible_file)
        mask = np.asarray(seg.dataobj) == 1

        return {
            'intensities': intensities.astype(np.int16),
            'mandible': mask,
            'spacing': np.array(
                self.spacing if self.pass_affine else img.header.get_zooms()
            ),
            'orientation': nibabel.io_orientation(
                np.eye(4) if self.pass_affine else img.affine,
            ),
            'affine': img.affine if self.pass_affine else np.eye(4),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_target(
        self,
        frac_file: Path,
    ) -> Dict[str, NDArray[np.int16]]:
        seg = nibabel.load(self.root / frac_file)
        labels = np.asarray(seg.dataobj)

        return {
            'labels': labels.astype(np.int16),
        }
