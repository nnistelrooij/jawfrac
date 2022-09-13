from pathlib import Path
import re
from typing import Any, Dict, List, Tuple, Union

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

    def _files(
        self,
        stage: str,
        exclude: List[str]=[
        ],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        scan_files = sorted(self.root.glob('**/image.nii.gz'))
        scan_files = [f for f in scan_files if re.search(self.filter, str(f))]
        scan_files = [f for f in scan_files if f.parent.name not in exclude]
        scan_files = [f.relative_to(self.root) for f in scan_files]

        if stage == 'predict':
            return scan_files

        ann_files = sorted(self.root.glob('**/seg.nii.gz'))
        ann_files = [f for f in ann_files if re.search(self.filter, str(f))]
        ann_files = [f for f in ann_files if f.parent.name not in exclude]
        ann_files = [f.relative_to(self.root) for f in ann_files]
        
        return list(zip(scan_files, ann_files))

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

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader(self.pred_dataset)
