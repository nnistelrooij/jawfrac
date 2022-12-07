from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
from scipy import ndimage

from jawfrac.data.transforms import MandibleCrop
from jawfrac.visualization import visualize


def wolla(path, threshold: int, min_voxels: int, dilate: int=0):
    img = nibabel.load(path)

    data = np.asarray(img.dataobj)
    affine = img.affine

    spacing = np.array(img.header.get_zooms())
    affine[:3, :3] /= spacing
    data = ndimage.zoom(data, zoom=spacing / 0.4)
    mask = data >= threshold

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate,
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


def wolla2(paths, threshold: int, min_voxels: int, dilate: int=0):
    img = nibabel.load(paths[0])
    img2 = nibabel.load(paths[1])
    img3 = nibabel.load(paths[2]) if len(paths) >= 3 else None

    
    spacing = np.array(img.header.get_zooms())
    affine = img.affine
    affine[:3, :3] /= spacing
    data = ndimage.zoom(np.asarray(img.dataobj), zoom=spacing / 0.4)
    mandible = ndimage.zoom(np.asarray(img2.dataobj) == 1, zoom=spacing / 0.4, output=float).round().astype(bool)

    mc = MandibleCrop(padding=10.0, extend=False)
    data_dict = {
        'intensities': data,
        'spacing': np.array([0.4, 0.4, 0.4]),
        'mandible': mandible,
    }
    if img3 is not None:
        data_dict['labels'] = np.asarray(img3.dataobj).astype(np.int16)
    data_dict = mc(**data_dict)


    data = data_dict['intensities']
    mask = data >= threshold

    if dilate:
        mask = ndimage.binary_dilation(
            input=mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilate,
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

    if 'labels' not in data_dict:
        return labels, points, colors
    else:
        return labels, points, colors, data_dict['labels']


def visualize_input(path):
    _, points, colors = wolla(path, 500, 20_000)
    points[:, 0] *= -1

    visualize(points, colors * 255)

def visualize_input_crop(path, path2):
    _, points, colors = wolla2([path, path2], 500, 20_000)
    points[:, 0] *= -1

    visualize(points, colors * 255)



def visualize_mandible(path, path2):
    labels, points, colors = wolla(path, 500, 20_000)
    points[:, 0] *= -1

    mandible_labels, mandible_points, mandible_colors = wolla(path2, 1, 1000, 1)
    mandible_points[:, 0] *= -1

    # mandible_colors[:, 0] = 1
    mandible_colors[:, 0] = 1 - mandible_colors[:, 0]
    mandible_colors[:, 1] = 0
    mandible_colors[:, 2] = 0

    both_labels = np.concatenate((
        np.column_stack(labels.nonzero()),
        np.column_stack(mandible_labels.nonzero()),
    ))
    both_labels, inverse, counts = np.unique(both_labels, axis=0, return_inverse=True, return_counts=True)

    labels = np.concatenate((
        labels[(counts == 1)[inverse[:labels.shape[0]]]],
        mandible_labels,
    ))
    points = np.concatenate((
        points[(counts == 1)[inverse[:points.shape[0]]]],
        mandible_points,
    ))
    colors = np.concatenate((
        colors[(counts == 1)[inverse[:colors.shape[0]]]],
        mandible_colors,
    ))
    

    visualize(points, colors * 255)

    return labels, points, colors

def visualize_mandible_fracture(path, path2, path3):
    labels, points, colors = visualize_mandible(path, path2)

    mandible_labels, mandible_points, mandible_colors = wolla(path3, 1, 1000, 0)
    # mandible_points[:, 0] *= -1

    # mandible_colors[:, 0] = 1
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

    points[:, 0] *= -1

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


def visualize_mandible_patch(path: Path, path2: Path):
    img = nibabel.load(path)
    data = np.asarray(img.dataobj)
    
    mandible = nibabel.load(path2)
    mandible = np.asarray(mandible.dataobj) >= 1
    
    # 17
    # data = ndimage.zoom(data, [1, 1, 2.5])
    # mandible = ndimage.zoom(mandible, [1, 1, 2.5], output=float).round().astype(bool)
    
    # 89
    data = ndimage.zoom(data, [1, 1, 2.5])
    mandible = ndimage.zoom(mandible, [1, 1, 2.5], output=float).round().astype(bool)
    data = data.clip(-1000, 3095)
    data = ((data + 1000) / 4095) * 255
    
    # adriane-widespread-goat
    # data = data.clip(-1000, 3095)
    # data = ((data + 1000) / 4095) * 255

    data = data.repeat(3).reshape(data.shape + (3,)).astype(float)
    mandible = mandible.nonzero()
    mandible = np.concatenate((
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 1),)),
        np.column_stack(mandible + (np.full(mandible[0].shape[:1], 2),)),
    ))
    data[tuple(mandible.T)] *= 0.5

    # 17
    out = np.full((154, 154, 3), 255)
    for i, idx in enumerate(range(188, 159, -4)):
        out[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 209:305, 246:342], ((1, 1), (1, 1), (0, 0)))

    # 89
    out = np.full((154, 154, 3), 255)
    for i, idx in enumerate(range(417, 388, -4)):
        out[
            8 * i:98 + 8 * i,
            56 - 8 * i:154 - 8 * i,
        ] = np.pad(data[idx, 179:275, 242:338], ((1, 1), (1, 1), (0, 0)))

    # adriane-widespread-goat
    # out = np.full((209, 209, 3), 255)
    # for i, idx in enumerate(range(99, 128, 4)):
    #     out[
    #         11 * i:132 + 11 * i,
    #         77 - 11 * i:209 - 11 * i,
    #     ] = np.pad(data[207:335, 230:358, idx], ((2, 2), (2, 2), (0, 0)))
    
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.savefig('/home/mkaailab/Downloads/mandible_patch.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def visualize_expand_label(scan_path, frac_path):
    img = nibabel.load(scan_path)
    shape = np.array(img.header.get_data_shape())
    intensities = np.asarray(img.dataobj)
    if intensities.min() == 0 and intensities.max() == 255:
        intensities = (intensities - intensities[intensities > 0].mean()) / 255 * 4096

    print('start')
    intensities = ndimage.zoom(
        input=intensities,
        zoom=np.array(img.header.get_zooms()) / 0.4,
    )
    print('interp 1')


    img = nibabel.load(frac_path)
    lbl = np.asarray(img.dataobj)
    label = ndimage.zoom(
        input=lbl,
        zoom=np.array(img.header.get_zooms()) / 0.4,
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
        zoom=0.4 / np.array(img.header.get_zooms()),
    ) >= 0.1


    # out = out.clip(0, 1)
    # out = np.tile(out[:, :, 59], (3, 1, 1)).transpose(1, 2, 0)
    # out[:, :, 1:] = 0
    # plt.imshow(out)
    # plt.axis('off')
    # plt.savefig('/mnt/d/Users/Niels-laptop/Documents/Master Thesis/expand_label_3.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    out = out[:shape[0], :shape[1], :shape[2]].astype(np.uint16)

    lbl[(lbl == 0) & (out == 1)] = 3

    img = nibabel.Nifti1Image(lbl, img.affine)
    nibabel.save(img, scan_path.parent / 'expand_label.nii.gz')



if __name__ == '__main__':
    path = Path('/home/mkaailab/Documents/fractures/Annotation UK')
    # path = Path(r'D:\')

    # visualize_input_crop(
    #     path / '89' / 'Patient89_main_image.nii.gz',
    #     path / '89' / 'mandible.nii.gz',
    # )
    # visualize_mandible(
    #     path / '89' / 'Patient89_main_image.nii.gz',
    #     path / '89' / 'mandible.nii.gz',
    # )
    # visualize_mandible_fracture(
    #     path / '52' / 'Patient52_main_image.nii.gz',
    #     path / '52' / 'mandible.nii.gz',
    #     path / '52' / 'frac_pred.nii.gz',
    # )

    visualize_mandible_patch(
        path / '89' / 'Patient89_main_image.nii.gz',
        path / '89' / 'frac_pred.nii.gz',
    )

    visualize_mandible_fracture(
        path / '112' / 'Patient112_main_image.nii.gz',
        path / '112' / 'mandible.nii.gz',
        path / '112' / 'frac_pred.nii.gz',
    )

    # visualize_patch(path)

    # visualize_input(
        # path / '112' / 'Patient112_main_image.nii.gz',
    # )

    visualize_mandible(
        path / '112' / 'Patient112_main_image.nii.gz',
        path / '112' / 'frac_pred.nii.gz'
    )
    # visualize_mandible_patch(path)

    visualize_expand_label(
        path / '114' / 'Patient114_main_image.nii.gz',
        path / '114' / 'label.nii.gz',
    )
