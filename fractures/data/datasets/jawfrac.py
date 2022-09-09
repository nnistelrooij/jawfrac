from pathlib import Path
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import NDArray

from miccai.data.datasets.base import MeshDataset
import fractures.data.transforms as T


class JawFracDataset(MeshDataset):
    """Dataset to load mandibular CT scans with fracture segmentations."""

    def __init__(
        self,
        mandible_crop_padding: float,
        regular_spacing: float,
        patch_size: int,
        stride: int,
        bone_hu_threshold: int,
        bone_volume_threshold: float,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.MandibleCrop(padding=mandible_crop_padding),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            T.BonePatchIndices(
                intensity_thresh=bone_hu_threshold,
                volume_thresh=bone_volume_threshold,
            ),
        )

        super().__init__(pre_transform=pre_transform, **kwargs)

    def load_scan(
        self,
        files: Path,
    ) -> Dict[str, NDArray[Any]]:
        scan_file, mandible_file = files

        img = nibabel.load(self.root / scan_file)
        intensities = np.asarray(img.dataobj)

        seg = nibabel.load(self.root / mandible_file)
        mask = np.asarray(seg.dataobj) == 1

        return {
            'intensities': intensities,
            'mandible': mask,
            'spacing': np.array(img.header.get_zooms()),
            'orientation': nibabel.io_orientation(img.affine),
            'shape': np.array(img.header.get_data_shape()),
        }

    def load_annotation(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.int16]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj, dtype=np.int16)

        return {
            'labels': labels,
        }
