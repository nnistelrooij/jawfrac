import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import open3d
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torchmetrics import ConfusionMatrix
from torchtyping import TensorType


def visualize(points: NDArray[np.float32], colors: NDArray[np.int64]) -> None:
    """Try to initialize window to visualize provided point cloud."""
    # intialize Open3D point cloud
    pcd = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(points),
    )
    pcd.colors = open3d.utility.Vector3dVector(colors / 255)

    # initialize Open3D window
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    # add each provided geometry to the Open3D window
    vis.add_geometry(pcd)

    # point size
    options = vis.get_render_option()
    options.point_size = 2

    # camera options
    view = vis.get_view_control()
    view.change_field_of_view(-55.0)  # defaut 60 - 55 = minimum 5
    view.set_zoom(0.66)
    view.set_front(np.array([[0., 1., 0.]]).T)
    view.set_up(np.array([[0., 0., 1.]]).T)
    
    # actually render the scene of geometries
    vis.run()


def draw_fracture_result(
    mandible: TensorType['D', 'H', 'W', torch.bool],
    pred: TensorType['D', 'H', 'W', torch.bool],
    target: TensorType['D', 'H', 'W', torch.bool],
    mandible_ratio: int=16,
) -> None:
    # determine TP, FP, and FN fracture voxel indices
    pred = pred.nonzero()
    target = target.nonzero()
    voxel_idxs = torch.cat((pred, target))

    unique, inverse, counts = torch.unique(
        voxel_idxs, return_inverse=True, return_counts=True, dim=0,
    )
    
    tp = unique[counts == 2]
    fp = pred[(counts == 1)[inverse][:pred.shape[0]]]
    fn = target[(counts == 1)[inverse][pred.shape[0]:]]

    # determine subsample of unique mandible voxel indices
    mandible = mandible.nonzero()
    voxel_idxs = torch.cat((mandible, unique))
    unique, inverse, counts = torch.unique(
        voxel_idxs, return_inverse=True, return_counts=True, dim=0,
    )

    mandible = mandible[(counts == 1)[inverse][:mandible.shape[0]]]
    mandible = mandible[torch.randperm(mandible.shape[0])]
    mandible = mandible[:mandible.shape[0] // mandible_ratio]

    # aggregate voxel indices and give colors
    voxel_idxs = torch.cat((mandible, tp, fp, fn)).float().cpu().numpy()
    colors = np.concatenate((
        np.tile([50, 50, 50], (mandible.shape[0], 1)),
        np.tile([100, 255, 100], (tp.shape[0], 1)),
        np.tile([255, 150, 100], (fp.shape[0], 1)),
        np.tile([255, 100, 200], (fn.shape[0], 1))
    ))
    
    visualize(voxel_idxs, colors)


def draw_positive_voxels(
    volume: TensorType['D', 'H', 'W', torch.bool],
) -> None:
    pos_voxel_idxs = volume.nonzero().float().cpu().numpy()
    colors = np.tile([50, 50, 50], (pos_voxel_idxs.shape[0], 1))

    visualize(pos_voxel_idxs, colors)


def draw_confusion_matrix(
    confmat: TensorType['C', 'C', torch.int64],
    title: str='Confusion Matrix',
) -> None:
    confmat = confmat.cpu().numpy()
    cmd = ConfusionMatrixDisplay(
        confusion_matrix=confmat,
        display_labels=('Controls', 'Fracture'),
    )
    cmd.plot()
    plt.title(title)
    plt.show()
