from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset


class MandibleSegDataModule(pl.LightningDataModule):
    """Implements data module that loads CBCT scans with annotated mandibles."""

    def __init__(
        self,
        root: Union[str, Path],
        regex_filter: str,
        val_size: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
        **dataset_cfg: Dict[str, Any],
    ):        
        super().__init__()
        
        self.root = Path(root)
        self.filter = regex_filter
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed =  seed
        self.dataset_cfg = dataset_cfg

    def _filter_files(
        self,
        pattern: str,
        exclude: List[str]=[],
    ) -> List[Path]:
        files = sorted(self.root.glob(pattern))
        files = [f for f in files if re.search(self.filter, str(f))]
        files = [f for f in files if f.parent.name not in exclude]
        files = [f.relative_to(self.root) for f in files]

        if not (self.root / 'Fabian overview.csv').exists():
            return files

        df = pd.read_csv(self.root / 'Fabian overview.csv')
        df = df.sort_values(by='Pseudonym')
        df = df[pd.isna(df['Note']) & ~pd.isna(df['Complete'])]
        df = df[~df['Complete'].str.match(r'.*[,+]')]
        files = [files[i] for i in df.index]

        return files

    def _files(
        self,
        stage: str,
        exclude: List[str]=[
        ],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        # scan_files = self._filter_files('**/image.nii.gz')
        scan_files = self._filter_files('**/*main*.nii.gz')

        if stage == 'predict':
            return scan_files[70:90]

        ann_files = self._filter_files('**/seg.nii.gz')
        
        return list(zip(scan_files, ann_files))

    def _split(
        self,
        files: List[Tuple[Path, Path]],
    ) -> Tuple[List[Tuple[Path, Path]]]:
        val_files = len(files) * self.val_size
        if val_files < 1:
            return files, []
        elif val_files > len(files) - 1:
            return [], files
    
        ss = ShuffleSplit(
            n_splits=1, test_size=self.val_size, random_state=self.seed,
        )
        train_idxs, val_idxs = next(ss.split(files))

        train_files = [files[i] for i in train_idxs]
        val_files = [files[i] for i in val_idxs]

        return train_files, val_files

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
