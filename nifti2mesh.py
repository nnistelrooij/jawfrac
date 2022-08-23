from pathlib import Path

import nibabel
import numpy as np
import open3d
from skimage.measure import marching_cubes

root = Path('/home/mka3dlab/Documents/fractures')

for main_file in root.glob('**/*image.nii.gz'):
    img = nibabel.load(main_file)
    img_data = img.get_fdata()


    seg_file = list(main_file.parent.glob('*segmentation.nii.gz'))[0]
    seg = nibabel.load(seg_file)
    seg_data = seg.get_fdata()

    for start_idx in range(seg_data.shape[-1]):
        frac = seg_data[..., start_idx:start_idx + 5].sum()
        if frac / np.prod(seg_data.shape[:-1]) > 0.005:
            print(frac)

    img_data_hist = np.sort(img_data.flatten())
    img_5perc = img_data_hist[int(0.05 * img_data_hist.shape[0])]
    img_95perc = img_data_hist[int(0.95 * img_data_hist.shape[0])]

    verts, faces, normals, values = marching_cubes(
        volume=img_data,
        level=700,
        spacing=img.header.get_zooms(),
    )

    mesh = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(verts),
        triangles=open3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    # mesh.vertex_normals = open3d.utility.Vector3dVector(normals)

    open3d.visualization.draw_geometries([mesh], width=1600, height=900)

    k = 4

