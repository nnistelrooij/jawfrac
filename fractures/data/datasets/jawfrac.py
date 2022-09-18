from pathlib import Path
from typing import Any, Dict, List

import nibabel
import numpy as np
from numpy.typing import NDArray

from miccai.data.datasets.base import MeshDataset
import fractures.data.transforms as T


class JawFracDataset(MeshDataset):
    """Dataset to load mandibular CT scans with fracture segmentations."""

    def __init__(
        self,
        stage: str,
        mandible_crop_padding: float,
        regular_spacing: float,
        patch_size: int,
        stride: int,
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
                T.ExpandLabel(bone_iters=1, all_iters=1),
            ) if stage == 'fit' else ()),
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_scan(
        self,
        files: List[Path],
    ) -> Dict[str, NDArray[Any]]:
        scan_file, mandible_file = files

        img = nibabel.load(self.root / scan_file)
        intensities = np.asarray(img.dataobj)

        seg = nibabel.load(self.root / mandible_file)
        mask = np.asarray(seg.dataobj) == 1

        print(scan_file.parent.stem)

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
