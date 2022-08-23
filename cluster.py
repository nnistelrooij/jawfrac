from pathlib import Path

import nibabel
import numpy as np
import open3d
from skimage.measure import marching_cubes
from sklearn.cluster import DBSCAN
from tqdm import tqdm


root = Path('/home/mka3dlab/Documents/fractures')

files = [Path('/home/mka3dlab/Documents/fractures/Annotation UK/49/Patient49_main_image.nii.gz')]
# files = root.glob('**/*image.nii.gz')
for main_file in tqdm(files):
    print(main_file)
    img = nibabel.load(main_file)
    img_data = img.get_fdata()
    img_data = np.clip(img_data, a_min=-100, a_max=1000)


    seg_file = list(main_file.parent.glob('*segmentation.nii.gz'))
    if not seg_file:
        continue

    seg_file = seg_file[0]
    seg = nibabel.load(seg_file)
    seg_data = seg.get_fdata()
    
    from skimage.measure import label

    seg_data_ = label(seg_data).astype(np.int16)
    counts = np.bincount(seg_data_.flatten())
    if np.any(counts < 10):
        min_count_idx = counts.argmin()
        seg_data_[seg_data_ == min_count_idx] = 0
        _, seg_data_inverse = np.unique(seg_data_, return_inverse=True)
        seg_data_ = seg_data_inverse.reshape(seg_data_.shape)
    print(seg_data_.max())

    img = nibabel.Nifti1Image(
        seg_data_.astype(np.int16),
        seg.affine,
        header=seg.header,
        extra=seg.extra,
        file_map=seg.file_map,
    )      

    nibabel.save(img, main_file.parent / f'label.nii.gz')


exit()



