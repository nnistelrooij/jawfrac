import json
from pathlib import Path
from typing import Any, Dict, Union

import nibabel
import numpy as np
from numpy.typing import NDArray

from fractures.data.datasets.base import CTDataset
from fractures.data.transforms import (
    Compose,
    Clip,
    IntervalNormalize,
    NaturalHeadPositionOrient,
    RegularScale,
)


class JawFracDataset(CTDataset):
    """Dataset to load mandibular CT scans with fracture segmentations."""

    MEAN = [2.0356, -0.6506, -90.0502]
    STD = 17.3281

    def __init__(
        self,
        **kwargs: Dict[str, Any],
    ) -> None:
        pre_transform = Compose(
            Clip(level=450.0, width=1100.0),
            IntervalNormalize(low=-1.0, high=1.0),
            # RegularScale(scale=4.0),
            # NaturalHeadPositionOrient(),
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
            'zooms': np.array(img.header.get_zooms()),
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
    
