from pathlib import Path

import nibabel
import numpy as np
import open3d


root = Path('/home/mka3dlab/Documents/jawfrac')


def bone_pointcloud(
    file: Path,
) -> open3d.geometry.PointCloud:
    img = nibabel.load(file)
    img_data = img.get_fdata()
    affine = img.affine[:-1, :-1]

    bone_mask = img_data > 1000
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

    return pcd


def label_pointcloud(
    file: Path,
) -> open3d.geometry.PointCloud:
    label = nibabel.load(file)
    label_data = label.get_fdata()
    affine = label.affine[:-1, :-1]

    label_mask = label_data > 0
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



img_files = list(root.glob('**/image.nii.gz'))
label_files = list(root.glob('**/label.nii.gz'))

prev_pcds = [bone_pointcloud(img_files[0]), label_pointcloud(label_files[0])]
for img_file, label_file in zip(img_files[1:], label_files[1:]):    
    bone_pcd = bone_pointcloud(img_file)
    label_pcd = label_pointcloud(label_file)

    # prev_pcds[0] = register(prev_pcds[0], bone_pcd)

    open3d.visualization.draw_geometries(prev_pcds + [bone_pcd, label_pcd])

    perv_pcds = [bone_pcd, label_pcd]
