from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
from scipy import ndimage

from jawfrac.visualization import visualize


def wolla(path, threshold: int, min_voxels: int, dilate: bool=False):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)
    affine = img.affine
    mask = data >= threshold

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=1,
        )

    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    _, inverse, counts = np.unique(labels.flatten(), return_inverse=True, return_counts=True)

    labels[(counts < min_voxels)[inverse].reshape(labels.shape)] = 0
    labels = labels[::-1]

    points = np.column_stack(labels.nonzero()).astype(float)
    hom_points = np.column_stack((points, np.ones_like(points[:, 0])))
    points = np.einsum('ij,kj->ki', affine, hom_points)
    points = points[:, :-1]

    colors = points[:, 0]
    colors = np.abs(colors - colors.mean())
    colors = (colors - colors.min()) / (colors.max() - colors.min())
    colors = 1 - np.tile(colors, (3, 1)).T

    return labels, points, colors


def visualize_input(path):
    _, points, colors = wolla(path, 130, 20_000)

    visualize(points, colors * 255)


def visualize_mandible(path, path2):
    labels, points, colors = wolla(path, 500, 20_000)
    mandible_labels, mandible_points, mandible_colors = wolla(path2, 1, 1000, True)
    mandible_colors[:, 0] = 1
    # mandible_colors[:, 0] = 1 - mandible_colors[:, 0]
    mandible_colors[:, 1:] = 0

    both_labels = np.concatenate((
        np.column_stack(labels.nonzero()),
        np.column_stack(mandible_labels.nonzero()),
    ))
    both_labels, inverse, counts = np.unique(both_labels, axis=0, return_inverse=True, return_counts=True)

    points = np.concatenate((
        points[(counts == 1)[inverse[:points.shape[0]]]],
        mandible_points,
    ))
    colors = np.concatenate((
        colors[(counts == 1)[inverse[:colors.shape[0]]]],
        mandible_colors,
    ))

    visualize(points, colors * 255)


def visualize_patch(path: Path):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    data = ndimage.zoom(data, [1, 1, 2.5])

    img = np.full((154, 154), 255)
    for i, idx in enumerate(range(188, 159, -4)):
        img[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 209:305, 246:342], (1, 1))
    
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(r'C:\Users\Niels-laptop\Documents\Master Thesis\patch.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_mandible_patch(path: Path, path2,):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    data = ndimage.zoom(data, [1, 1, 2.5])
    
    mandible = nibabel.load(path2)

    mandible = np.asarray(mandible.dataobj)
    mandible = ndimage.zoom(mandible, [1, 1, 2.5], output=float).round().astype(bool)

    data = data.repeat(3).reshape(data.shape + (3,)).astype(float)
    mandible = mandible.nonzero()
    mandible = np.concatenate((
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 1),)),
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 2),)),
    ))
    data[tuple(mandible.T)] *= 0.5

    out = np.full((154, 154, 3), 255)
    for i, idx in enumerate(range(188, 159, -4)):
        out[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 209:305, 246:342], ((1, 1), (1, 1), (0, 0)))
    
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.savefig(r'C:\Users\Niels-laptop\Documents\Master Thesis\mandible_patch.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_expand_label():
    root = Path('/mnt/d/Users/Niels-laptop/Downloads/fractures/')

    scan_file = root / 'Patient17_main_image.nii.gz'
    frac_file = root / 'Patient17_seg_normal.nii.gz'

    img = nibabel.load(scan_file)
    intensities = np.asarray(img.dataobj)
    intensities = (intensities - intensities[intensities > 0].mean()) / 255 * 4096

    print('start')
    intensities = ndimage.zoom(
        input=intensities,
        zoom=[1, 1, 2.5],
    )
    print('interp 1')


    img = nibabel.load(frac_file)
    label = np.asarray(img.dataobj)
    label = ndimage.zoom(
        input=label,
        zoom=[1, 1, 2.5],
        output=float,
    ).round().astype(bool)
    print('interp 2')

    out = ndimage.binary_dilation(
        input=label,
        structure=ndimage.generate_binary_structure(3, 2),
        iterations=1,
        mask=intensities >= 300
    )
    out = ndimage.binary_dilation(
        input=out,
        structure=ndimage.generate_binary_structure(3, 2),
        iterations=1,
    ).astype(np.float32)

    out = ndimage.gaussian_filter(
        input=out.astype(np.float32),
        sigma=0.5,
    )

    out = ndimage.zoom(
        input=out,
        zoom=[1, 1, 0.4],
    ) >= 0.2


    # out = out.clip(0, 1)
    # out = np.tile(out[:, :, 59], (3, 1, 1)).transpose(1, 2, 0)
    # out[:, :, 1:] = 0
    # plt.imshow(out)
    # plt.axis('off')
    # plt.savefig('/mnt/d/Users/Niels-laptop/Documents/Master Thesis/expand_label_3.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    out = out[:, :, :199].astype(np.uint16)

    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, root / 'expand_label_bi.nii.gz')



if __name__ == '__main__':
    path = Path('/mnt/d/Users/Niels-laptop/Documents/Annotation UK/103/Patient103_main_image.nii.gz')
    # path = Path(r'D:\')
    # visualize_input(path)
    # visualize_patch(path)

    visualize_mandible(path, path.parent / 'mandible200.nii.gz')
    # visualize_mandible_patch(path)

    # visualize_mandible_patch(path, path.parent / 'frac_pred_linear.nii.gz')

    # visualize_expand_label()
