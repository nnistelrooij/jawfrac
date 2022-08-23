from pathlib import Path
import pickle
from typing import Any

from torch.utils.data import Dataset


class DatasetCache(dict):
    """Implements cache to load and store preprocessed dataset from storage."""

    def __init__(
        self,
        dataset: Dataset,
        cache_path: Path,
    ) -> None:
        super().__init__()

        if cache_path.exists():
            print(f'Loading dataset from {cache_path}.')
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)

            for key, value in cache.items():
                super().__setitem__(key, value)
        
        self.dataset = dataset
        self.cache_path = cache_path

    def __setitem__(self, key: int, value: Any) -> None:
        super().__setitem__(key, value)

        if (
            not self.cache_path.exists()
            and len(self) == len(self.dataset)
        ):
            print(f'Storing dataset as {self.cache_path}.')
            with open(self.cache_path, 'wb') as f:
                pickle.dump(dict(self), f)
