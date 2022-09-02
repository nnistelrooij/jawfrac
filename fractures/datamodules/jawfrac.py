from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
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

    def _files(
        self,
        stage: str,
        exclude: List[str]=[],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        scan_files = sorted(self.root.glob('**/image.nii.gz'))
        scan_files = [f for f in scan_files if re.search(self.filter, str(f))]
        scan_files = [f for f in scan_files if f.parent.name not in exclude]
        scan_files = [f.relative_to(self.root) for f in scan_files]

        jaw_files = sorted(self.root.glob('**/mandible.nii.gz'))
        jaw_files = [f for f in jaw_files if re.search(self.filter, str(f))]
        jaw_files = [f for f in jaw_files if f.parent.name not in exclude]
        jaw_files = [f.relative_to(self.root) for f in jaw_files]

        if stage == 'predict':
            return list(zip(scan_files, jaw_files))

        frac_files = sorted(self.root.glob('**/fractures.nii.gz'))
        frac_files = [f for f in frac_files if re.search(self.filter, str(f))]
        frac_files = [f for f in frac_files if f.parent.name not in exclude]
        frac_files = [f.relative_to(self.root) for f in frac_files]
        
        return list(zip(zip(scan_files, jaw_files), frac_files))

    def _split(
        self,
        files: List[Tuple[Path, Path]],
    ) -> Tuple[List[Tuple[Path, Path]]]:
        if self.val_size == 0.0:
            return files, []
        elif self.val_size == 1.0:
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
        shuffle: bool=False,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.predict_dataset)
