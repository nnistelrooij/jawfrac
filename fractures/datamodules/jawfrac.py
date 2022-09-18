from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset


class JawFracDataModule(pl.LightningDataModule):
    """Implements data module that loads volumes of mandibles."""

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
        **dataset_cfg: Dict[str, Any]
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
        exclude: List[str]=['12', '103', '123', '125', '128', '130', '148', '155', '171', '186'],
    ) -> List[Path]:
        files = sorted(self.root.glob(pattern))
        files = [f for f in files if re.search(self.filter, str(f))]
        files = [f for f in files if f.parent.name not in exclude]
        files = [f.relative_to(self.root) for f in files]

        if not (self.root / 'Sophie overview.csv').exists():
            return files

        df = pd.read_csv(self.root / 'Sophie overview.csv')
        mask = pd.isna(df['HU']) & pd.isna(df['Dislocated'])
        idxs = mask.index[mask]

        patients = list(map(lambda p: int(p.parent.stem), files))
        files = [f for f, p in zip(files, patients) if p - 1 in idxs]

        return files

    def _files(
        self,
        stage: str,
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        scan_files = self._filter_files('**/*main*.nii.gz')
        jaw_files = self._filter_files('**/mandible.nii.gz')

        if stage == 'predict':
            return list(zip(scan_files, jaw_files))[:5]

        frac_files = self._filter_files('**/Patient*segm*.nii.gz')
        frac_files += self._filter_files('**/*seg_normal*.nii.gz')
        frac_files += self._filter_files('**/Seg*.nii.gz')

        files = list(zip(zip(scan_files, jaw_files), sorted(frac_files)))

        if self.trainer.state.stage == RunningStage.TESTING:
            _, files = self._split(files)
        
        return files

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
