from typing import Sequence
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torchtyping import TensorType
from tqdm import tqdm

import jawfrac.data.transforms as T
from jawfrac.metrics import FracPrecision, FracRecall


def compute_times(path: Path):
    df = pd.read_csv(path)

    times = pd.to_timedelta(df['T3'])

    p_avg, p_std = times[has_fracture].mean(), times[has_fracture].std()
    print(f'Patient time: {p_avg} +- {p_std}.')

    c_avg, c_std = times[~has_fracture].mean(), times[~has_fracture].std()
    print(f'Control time: {c_avg} +- {c_std}.')

    t_avg, t_std = times.mean(), times.std()
    print(f'Total time: {t_avg} +- {t_std}.')


def confusion_matrix(
    path: Path,
    has_fracture: Sequence[bool],
    verbose=False,
):
    df = pd.read_csv(path)

    radiologist_fracture = df['Fracture'].isna().tolist()

    cmd = ConfusionMatrixDisplay.from_predictions(
        has_fracture,
        radiologist_fracture,
        display_labels=['Control', 'Fracture'],
    )
    cmd.plot()

    if verbose:
        plt.show()


def combine_annotations(
    root: Path,
    transform=T.Compose(
        T.NonNegativeCrop(),
        T.RegularSpacing(spacing=0.4),
        T.NaturalHeadPositionOrient(),
    ),
) -> TensorType['D', 'H', 'W', torch.bool]:
    scan_file = root / 'scan.nii.gz'

    img = nibabel.load(scan_file)
    intensities = np.asarray(img.dataobj)

    # convert 8-bit to 12-bit
    if intensities.min() == 0 and intensities.max() == 255:
        center = intensities[intensities > 0].mean()
        intensities = (intensities - center) / 255 * 4095

    # clip intensities to sensible range
    intensities = intensities.clip(-1024, 3096)

    annotations = []
    seg_files = [f for f in root.glob('*') if 'scan.nii.gz' not in str(f)]
    seg_files = [f for f in seg_files if 'original' not in str(f)]
    for seg_file in seg_files:
        img = nibabel.load(seg_file)
        data = np.asarray(img.dataobj).astype(np.int16)
        data_dict = {
            'intensities': intensities,
            'labels': data,
            'spacing': img.header.get_zooms(),
            'orientation': nibabel.io_orientation(img.affine),
        }
        data = transform(**data_dict)['labels']
        
        idxs = np.column_stack(np.nonzero(data))
        centroid = idxs.mean(axis=0)

        centroid_voxel = centroid.round().astype(int)
        annotation = np.zeros(data.shape, dtype=bool)
        annotation[tuple(centroid_voxel)] = True

        annotations.append(annotation)

    annotations = np.stack(annotations)
    combined_annotation = np.any(annotations, axis=0)

    return torch.tensor(combined_annotation)


def expand_label(
    root: Path,
    transform=T.Compose(
        T.NonNegativeCrop(),
        T.RegularSpacing(spacing=0.4),
        T.NaturalHeadPositionOrient(),
        T.ExpandLabel(bone_iters=1, all_iters=1, negative_iters=16, smooth=0.5),
    ),
) -> TensorType['D', 'H', 'W', torch.bool]:
    scan_file = list(root.glob('*main*'))[0]
    img = nibabel.load(scan_file)
    intensities = np.asarray(img.dataobj)

    # convert 8-bit to 12-bit
    if intensities.min() == 0 and intensities.max() == 255:
        center = intensities[intensities > 0].mean()
        intensities = (intensities - center) / 255 * 4095

    # clip intensities to sensible range
    intensities = intensities.clip(-1024, 3096)

    label_file = root / 'label.nii.gz'
    img = nibabel.load(label_file)
    data = np.asarray(img.dataobj).astype(np.int16)

    data_dict = {
        'intensities': intensities,
        'labels': data,
        'spacing': img.header.get_zooms(),
        'orientation': nibabel.io_orientation(img.affine),
    }
    data = transform(**data_dict)['labels']

    label = data >= 0.1

    return torch.tensor(label)


root = Path('/mnt/diag/fractures')

with open(root / 'Radiologists' / 'key.txt', 'r') as f:
    key = [l.strip() for l in f.readlines() if l.strip()]

has_fracture = pd.Series(['Annotation' in f for f in key])


precision_metric = FracPrecision(voxel_thresh=(1, 1000))
recall_metric = FracRecall(voxel_thresh=(1, 1000))

for radiologist in [
    'Pieter van Lierop',
]:
    print(radiologist)

    radiologist_root = root / 'Radiologists' / radiologist
    compute_times(radiologist_root / 'times.csv')
    confusion_matrix(radiologist_root / 'times.csv', has_fracture)

    tqdm_iter = tqdm(key)
    for i, path in enumerate(tqdm_iter, 1):
        if 'Controls' in path:
            continue

        tqdm_iter.set_description(path)

        seg = combine_annotations(radiologist_root / f'scan{i}')
        label = expand_label((root / path).parent)

        precision_metric(seg, label)
        recall_metric(seg, label)
    
    print(f'Precision: {precision_metric.compute()}')
    print(f'Recall: {recall_metric.compute()}')

