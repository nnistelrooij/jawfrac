from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Tuple

import pytorch_lightning as pl
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset


class VolumeDataModule(pl.LightningDataModule):
    """Implements data module that loads 3D volumes with intensity values."""

    def __init__(
        self,
        exclude: List[str],
        regex_filter: str,
        val_size: float,
        test_size: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
        **dataset_cfg: Dict[str, Any],
    ):        
        super().__init__()
        
        self.root = Path(dataset_cfg['root'])
        self.exclude = '.*/' + '/.*|.*/'.join(exclude) + '/.*'
        self.filter = regex_filter
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed =  seed
        self.dataset_cfg = dataset_cfg

    def _filter_files(
        self,
        pattern: str,
    ) -> List[Path]:
        files = sorted(self.root.glob(pattern))
        files = [f for f in files if re.search(self.filter, str(f))]
        files = [f for f in files if re.search(self.exclude, str(f)) is None]
        files = [f.relative_to(self.root) for f in files]

        return files

    def _split(
        self,
        files: List[Tuple[Path, ...]],
    ) -> Tuple[
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
        List[Tuple[Path, ...]],
    ]:
        val_files = len(files) * self.val_size
        test_files = len(files) * self.test_size
        if val_files < 1 and test_files < 1:
            return files, [], []
        elif val_files > len(files) - 1:
            return [], files, []
        elif test_files > len(files) - 1:
            return [], [], files
    
        ss = ShuffleSplit(
            n_splits=1,
            test_size=self.val_size + self.test_size,
            random_state=self.seed,
        )
        train_idxs, val_test_idxs = next(ss.split(files))

        train_files = [files[i] for i in train_idxs]
        val_test_files = [files[i] for i in val_test_idxs]

        if val_files > len(val_test_files) - 1:
            return train_files, val_test_files, [],
        elif test_files > len(val_test_files) - 1:
            return train_files, [], val_test_files
    
        ss = ShuffleSplit(
            n_splits=1,
            test_size=self.test_size / (self.val_size + self.test_size),
            random_state=self.seed,
        )
        val_idxs, test_idxs = next(ss.split(val_test_files))

        val_files = [val_test_files[i] for i in val_idxs]
        test_files = [val_test_files[i] for i in test_idxs]

        return train_files, val_files, test_files

    def _dataloader(
        self,
        dataset: Dataset,
        collate_fn: Callable[..., Dict[str, Any]],
        shuffle: bool=False,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(
            self.train_dataset, self.fit_collate_fn, shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, self.fit_collate_fn)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, self.test_collate_fn)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.predict_dataset, self.predict_collate_fn)
