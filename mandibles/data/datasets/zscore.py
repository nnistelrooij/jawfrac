from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import os, sys
sys.path.append(os.getcwd())

from mandibles.datamodules import MandibleSemSegDataModule


def mesh_means_stds(
    dataloader: DataLoader,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    total_means, total_sq_means = np.zeros((2, 3))
    total_count = 0

    for batch in tqdm(iter(dataloader), total=len(dataloader)):
        x, _, _ = batch

        points = x.C.numpy()
        means = points.mean(axis=0)
        sq_means = (points ** 2).mean(axis=0)
        count = points.shape[0]

        total_count += count
        total_means += (count / total_count) * (means - total_means)
        total_sq_means += (count / total_count) * (sq_means - total_sq_means)

    total_stds = np.sqrt(total_sq_means - total_means ** 2)
    
    return total_means, total_stds


if __name__ == '__main__':
    with open('mandibles/config/semseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dm = MandibleSemSegDataModule(seed=config['version'], **config['datamodule'])
    dm.setup(stage='predict')

    means, stds = mesh_means_stds(dm.predict_dataloader())
    print('Means:', means)
    print('STDs:', stds)
