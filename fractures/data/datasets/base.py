from hashlib import blake2s
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from fractures.data.datasets.cache import DatasetCache


class CTDataset(Dataset):
    """Dataset to load CT scans with voxel segmentations."""

    def __init__(
        self,
        stage: str,
        root: Path,
        files: Union[List[Path], List[Tuple[Path, Path]]],
        pre_transform: Callable[..., Dict[str, Any]]=dict,
        transform: Callable[..., Dict[str, Any]]=dict,
    ) -> None:
        super().__init__()

        self.stage = stage
        self.root = root
        self.files = files
        self.pre_transform = pre_transform
        self.transform = transform
        self.cache = DatasetCache(
            dataset=self,
            cache_path=(root / str(hash(self))).with_suffix('.pkl'),
        ) if False else {}

    def load_scan(self, file: Path):
        raise NotImplementedError

    def load_annotation(self, file: Path):
        raise NotImplementedError

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Union[NDArray[Any], int]]:
        # load data from cache or storage
        if index in self.cache:
            data_dict = self.cache[index]
        else:
            if self.stage == 'predict':
                scan_file = self.files[index]
                data_dict = self.load_scan(scan_file)
            else:
                scan_file, ann_file = self.files[index]
                data_dict = {
                    **self.load_scan(scan_file),
                    **self.load_annotation(ann_file),
                }
            
            # apply preprocessing transformations and cache
            data_dict = self.pre_transform(**data_dict)
            self.cache[index] = data_dict

        # apply data transformations
        data_dict = self.transform(**data_dict)

        return data_dict

    def __hash__(self) -> int:
        h = blake2s()
        h.update(repr(self.files).encode('utf-8'))
        h.update(repr(self.pre_transform).encode('utf-8'))
        return int(h.hexdigest(), base=16)

    def __len__(self) -> int:
        return len(self.files)
