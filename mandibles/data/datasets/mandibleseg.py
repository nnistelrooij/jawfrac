from pathlib import Path
from typing import Any, Dict

import nibabel
import numpy as np
from numpy.typing import NDArray

from miccai.data.datasets.base import MeshDataset
import mandibles.data.transforms as T


class MandiblePatchSegDataset(MeshDataset):
    """Dataset to load mandibular CT scans with fracture segmentations."""

    def __init__(
        self,
        stage: str,
        crop_padding: float,
        regular_spacing: float,
        patch_size: int,
        stride: int,
        pos_volume_threshold: float,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = T.Compose(
            T.NonNegativeCrop(padding=crop_padding),
            T.RegularSpacing(spacing=regular_spacing),
            T.NaturalHeadPositionOrient(),
            T.PatchIndices(patch_size=patch_size, stride=stride),
            T.PositiveNegativePatchIndices(volume_thresh=pos_volume_threshold)
            if stage == 'fit' else dict,
        )

        super().__init__(stage=stage, pre_transform=pre_transform, **kwargs)

    def load_scan(
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

    def load_annotation(
        self,
        file: Path,
    ) -> Dict[str, NDArray[np.uint16]]:
        seg = nibabel.load(self.root / file)
        labels = np.asarray(seg.dataobj)

        return {
            'labels': (labels == 2).astype(np.int16),
        }
