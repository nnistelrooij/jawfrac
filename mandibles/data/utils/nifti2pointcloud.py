from pathlib import Path

import nibabel
import numpy as np
import open3d
import pymeshlab
from scipy.spatial.transform import Rotation
from scipy import ndimage


def bone_pointcloud(
    file: Path,
) -> open3d.geometry.PointCloud:
    img = nibabel.load(file)
    img_data = img.get_fdata()
    affine = img.affine
    # r = Rotation.from_euler('yzx', [-90, 180, 90], degrees=True)
    # affine = r.as_matrix()
    rot = np.eye(4)
    r = Rotation.from_euler('yx', [180, 90], degrees=True)
    rot[:-1, :-1] = r.as_matrix()
    affine = affine @ rot
    affine = affine[:-1, :-1]

    bone_mask = img_data >= 500

    labels, _ = ndimage.label(bone_mask)
    _, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True,
    )

    small_components = counts < 1000
    bone_mask[small_components[inverse.reshape(bone_mask.shape)]] = False

    bone_idxs = np.column_stack(bone_mask.nonzero())
    # bone_idxs = np.column_stack(
    #     (bone_idxs, np.ones(bone_idxs.shape[0])),
    # )
    bone_idxs = (np.einsum('ij,kj->ki', affine, bone_idxs))


    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(bone_idxs),
    )
    pcd.colors = open3d.utility.Vector3dVector(
        np.zeros((bone_idxs.shape[0], 3)),
    )

    return bone_mask, pcd


def label_pointcloud(
    img_data,
    file: Path,
) -> open3d.geometry.PointCloud:
    label = nibabel.load(file)
    label_data = label.get_fdata()

    label_data_new = ndimage.binary_dilation(
        input=label_data == 2,
        structure=ndimage.generate_binary_structure(3, 3),
        iterations=2,
        mask=img_data & (label_data == 0),
    )

    affine = label.affine
    rot = np.eye(4)
    r = Rotation.from_euler('yx', [180, 90], degrees=True)
    rot[:-1, :-1] = r.as_matrix()
    affine = affine @ rot
    affine = affine[:-1, :-1]

    # r = Rotation.from_euler('yzx', [-90, 180, 90], degrees=True)
    
    # r = Rotation.from_euler('yx', [180, 90], degrees=True)
    # affine = r.as_matrix()

    label_mask = label_data_new & img_data
    label_idxs = np.column_stack(label_mask.nonzero())
    # label_idxs = np.column_stack(
    #     (label_idxs, np.ones(label_idxs.shape[0])),
    # )
    label_idxs = (np.einsum('ij,kj->ki', affine, label_idxs))

    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(label_idxs),
    )
    pcd.colors = open3d.utility.Vector3dVector(
        np.tile(np.array([[1, 0, 0]]), (label_idxs.shape[0], 1)),   
    )

    return label_data_new, affine, pcd

def clean_bone_pointcloud(img_data, label_data, affine):
    mask = img_data & ~label_data
    labels, _ = ndimage.label(mask)
    _, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True,
    )

    small_components = counts < 1000
    img_data[small_components[inverse.reshape(mask.shape)]] = False

    bone_idxs = np.column_stack(img_data.nonzero())
    # bone_idxs = np.column_stack(
    #     (bone_idxs, np.ones(bone_idxs.shape[0])),
    # )
    bone_idxs = (np.einsum('ij,kj->ki', affine, bone_idxs))


    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(bone_idxs),
    )
    pcd.colors = open3d.utility.Vector3dVector(
        np.zeros((bone_idxs.shape[0], 3)),
    )

    return pcd

def register(pcd1, pcd2):
    points1 = np.asarray(pcd1.points)
    pcd1.points = open3d.utility.Vector3dVector(points1 - points1.mean(0))
    points2 = np.asarray(pcd2.points)
    pcd2.points = open3d.utility.Vector3dVector(points2 - points2.mean(0))

    reg_p2p = open3d.pipelines.registration.registration_icp(
        pcd1, pcd2, 0.1, np.eye(4),
        open3d.pipelines.registration.TransformationEstimationPointToPoint(),
        open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    affine = reg_p2p.transformation
    points1 = np.column_stack(
        (points1, np.ones(points1.shape[0]))
    )
    points1 = np.einsum('ij,kj->ki', affine, points1)[:, :3]
    pcd1.points = open3d.utility.Vector3dVector(points1)

    return pcd1



root = Path('/mnt/diag/nielsvannistelrooij/Fabian_clean')
# root = Path('/home/mka3dlab/Documents/fractures/jawfrac')

img_files = sorted(root.glob('**/image.nii.gz'))
label_files = sorted(root.glob('**/seg.nii.gz'))

for img_file, label_file in zip(img_files, label_files):
    print(img_file.parent.stem) 
    img_data, bone_pcd = bone_pointcloud(img_file)
    label_data, affine, label_pcd = label_pointcloud(img_data, label_file)
    bone_pcd = clean_bone_pointcloud(img_data, label_data, affine)


    # prev_pcds[0] = register(prev_pcds[0], bone_pcd)

    open3d.visualization.draw_geometries([bone_pcd, label_pcd])

    perv_pcds = [bone_pcd, label_pcd]
