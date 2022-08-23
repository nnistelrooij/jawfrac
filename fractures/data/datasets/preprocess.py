import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())

import nibabel
import numpy as np
from tqdm import tqdm

from fractures.data.transforms import (
    Compose,
    NaturalHeadPositionOrient,
    RegularScale,
)


def preprocess(root: Path, scale=4.0):
    transform = Compose(
        RegularScale(scale),
        NaturalHeadPositionOrient(),
    )

    img_files = sorted(root.glob('**/image.nii.gz'))
    label_files = sorted(root.glob('**/label.nii.gz'))

    for img_file, label_file in zip(tqdm(img_files), label_files):
        img = nibabel.load(img_file)
        label = nibabel.load(label_file)

        data_dict = transform(**{
            'intensities': np.asarray(img.dataobj),
            'labels': np.asarray(label.dataobj),
            'affine': img.affine,
            'zooms': np.array(img.header.get_zooms()),
        })

        img = nibabel.Nifti1Image(data_dict['intensities'], np.eye(4), img.header)
        img.header.set_zooms((scale, scale, scale))
        nibabel.save(img, img_file.parent / 'test4d.nii.gz')

        label = nibabel.Nifti1Image(data_dict['labels'], np.eye(4), label.header)
        label.header.set_zooms((scale, scale, scale))
        nibabel.save(label, label_file.parent / 'label4d.nii.gz')



if __name__ == '__main__':
    root = Path('/home/mka3dlab/Documents/jawfrac')
    preprocess(root)
