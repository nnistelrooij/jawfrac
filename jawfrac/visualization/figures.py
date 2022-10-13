from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
from numpy.typing import NDArray
import open3d
from scipy import ndimage
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import ConfusionMatrix

from jawfrac.visualization.open3d import visualize


def draw_confusion_matrix(
    confmat: ConfusionMatrix,
) -> None:
    confmat = confmat.compute().cpu().numpy()
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=confmat,
        display_labels=('Controls', 'Fracture'),
    )
    cmd.plot()
    plt.show()


def wolla(path, threshold: int, min_voxels: int):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)
    affine = img.affine
    mask = data >= threshold

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
    labels, points, colors = wolla(path, 130, 20_000)
    mandible_labels, mandible_points, mandible_colors = wolla(path2, 1, 2000)
    mandible_colors[:, 0] = 1 - mandible_colors[:, 0]
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


def visualize_mandible_patch(path: Path):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)

    data = ndimage.zoom(data, [1, 1, 2.5])
    
    mandible = nibabel.load(path.parent / 'mandible3.nii.gz')

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


if __name__ == '__main__':
    path = Path(r'C:\Users\Niels-laptop\Downloads\fractures\Patient17_main_image.nii.gz')
    # visualize_input(path)
    # visualize_patch(path)

    # visualize_mandible(path, path.parent / 'mandible3.nii.gz')
    # visualize_mandible_patch(path)

    visualize_mandible(path, path.parent / 'frac_pred.nii.gz')
