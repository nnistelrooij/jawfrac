import copy
from multiprocessing import cpu_count, Pool
import os
from pathlib import Path
from typing import Any, List, Tuple

import nibabel
import numpy as np
from numpy.typing import NDArray
import pydicom
from tqdm import tqdm


class DICOM:

    def __init__(
        self,
        spacing: List[float],
        position: List[float],
        orientation: List[float],
        content: NDArray[np.float32],
    ):
        self.spacing = spacing
        self.position = position
        self.orientation = orientation
        self.shape = content.shape
        self.content = content

    @property
    def affine(self) -> NDArray[np.float32]:
        F11, F21, F31 = self.orientation[3:]
        F12, F22, F32 = self.orientation[:3]

        slices = self.shape[0]

        dr, dc = self.spacing[:2]
        pos1 = self.position[:3]
        posN = self.position[3:]

        return np.array([
            [F11 * dr, F12 * dc, (posN[0] - pos1[0]) / (slices - 1), pos1[0]],
            [F21 * dr, F22 * dc, (posN[1] - pos1[1]) / (slices - 1), pos1[1]],
            [F31 * dr, F32 * dc, (posN[2] - pos1[2]) / (slices - 1), pos1[2]],
            [0, 0, 0, 1]
        ])
    
    def to_numpy(self) -> NDArray[np.int16]:
        return self.content.astype(np.int16)

    @staticmethod
    def rescale_intensities(ds: pydicom.Dataset) -> NDArray[Any]:
        # unit = getattr(ds, 'RescaleType', 'HU')

        # assert unit == 'HU', f'Must have Hounsfield unit, got {unit}.'

        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        slope = getattr(ds, 'RescaleSlope', 1.0)

        return ds.pixel_array * slope + intercept

    @staticmethod
    def voxel_spacing(ds: pydicom.Dataset) -> List[float]:
        in_plane = getattr(ds, 'PixelSpacing', None)
        slice_thickness = getattr(ds, 'SliceThickness', None)

        if in_plane is None or slice_thickness is None:
            ds = ds[0x5200, 0x9229][0]
            ds = ds[0x0028, 0x9110][0]

            in_plane = ds.PixelSpacing
            slice_thickness = ds.SliceThickness    

        spacing = [slice_thickness] + list(in_plane)
        spacing = spacing[::-1]
        spacing = [float(s) for s in spacing]
        spacing = [0.402 for _ in range(3)]

        return spacing

    @staticmethod
    def voxel_position(ds: pydicom.Dataset) -> List[float]:
        pos1 = getattr(ds, 'ImagePositionPatient', None)
        posN = None

        if pos1 is None and (0x5200, 0x9230) in ds:
            slices = ds[0x5200, 0x9230]
            position = []
            for idx in [0, -1]:
                if hasattr(slices[idx], 'ImagePositionPatient'):
                    coords = slices[idx].ImagePositionPatient
                else:
                    ds = slices[idx][0x0020, 0x9113][0]
                    coords = ds.ImagePositionPatient

                position.append(coords)

            pos1, posN = position

        if pos1 is None:
            pos1 = [0, 0, 0]

        if posN is None:
            posN = copy.copy(pos1)
            posN[2] -= (ds.NumberOfFrames - 1) * ds.SliceThickness
            
        position = [float(c) for c in list(pos1) + list(posN)]

        return position

    @staticmethod
    def voxel_orientation(ds: pydicom.Dataset) -> List[float]:
        orientation = getattr(ds, 'ImageOrientationPatient', None)

        if orientation is None:
            slices = ds[0x5200, 0x9230]
            orientation = getattr(slices[0], 'ImageOrientationPatient', None)

        if orientation is None and (0x0020, 0x9116) in slices[0]:
            de = slices[0][0x0020, 0x9116][0]
            orientation = getattr(de, 'ImageOrientationPatient', None)

        if orientation is None:
            ds = ds[0x5200, 0x9229][0]
            ds = ds[0x0020, 0x9116][0]
            orientation = ds.ImageOrientationPatient

        orientation = [float(s) for s in orientation]

        return orientation

    @staticmethod
    def read_file(path: Path):
        ds = pydicom.read_file(path)

        image = DICOM.rescale_intensities(ds)
        spacing = DICOM.voxel_spacing(ds)
        orientation = DICOM.voxel_orientation(ds)
        position = DICOM.voxel_position(ds)

        return DICOM(spacing, position, orientation, image)


class PSG:

    def __init__(
        self,
        version: int,
        shape: Tuple[int, int, int],
        obj_type: str,
        obj_id: str,
        content: NDArray[np.bool_],
    ):
        self.version = version
        self.shape = shape
        self.type = obj_type
        self.id = obj_id
        self.content = content

        assert version == 2

    def to_numpy(self) -> NDArray[np.uint16]:
        out = self.content.reshape(self.shape)
        out = out.astype(np.uint16)
        out = out.transpose(2, 1, 0)[::-1]

        return out

    @staticmethod
    def read_file(path: Path):
        fp = open(path, 'rb')

        version = int.from_bytes(fp.read(1), byteorder='big')

        shape = ()
        for _ in range(3):
            dim = int.from_bytes(fp.read(2), byteorder='big')
            shape = shape + (dim,)

        object_props = []
        for _ in range(2):
            str_size = int.from_bytes(fp.read(1), byteorder='big')
            str_encoded = fp.read(str_size)
            string = str_encoded.decode('utf-8')
            object_props.append(string)

        object_type, object_id = object_props

        out = np.zeros(np.prod(shape), dtype=bool)
        while True:
            chunk_start = fp.read(4)
            if chunk_start:
                start = int.from_bytes(chunk_start, byteorder='big')
            else:
                break

            chunk_end = fp.read(4)
            if chunk_end:
                end = int.from_bytes(chunk_end, byteorder='big')
                out[start:end] = True
            else:
                out[start:] = True
                break
        
        fp.close()

        return PSG(version, shape, object_type, object_id, out)


def write_nifti(
    img: NDArray[Any],
    filename: str,
    affine: NDArray[np.float32],
) -> None:
    img = nibabel.Nifti1Image(img, affine)
    nibabel.save(img, filename)


def convert_case(dir_path: Path) -> None:
    scan_dir = dir_path / 'scan'
    dcm_files = list(scan_dir.glob('*.dcm'))

    assert len(dcm_files) == 1

    try:
        dicom = DICOM.read_file(dcm_files[0])
    except (AssertionError, AttributeError, TypeError, ValueError) as e:
        print(f'Failed: {dir_path}')
        print(e)
        return dir_path

    # set any annotated voxel to 1
    psg_dir = dir_path / 'psg_manual_ann'
    psg_files = psg_dir.glob('**/*.psg')

    seg = np.zeros(dicom.shape, dtype=np.uint16)
    for psg_file in psg_files:
        psg = PSG.read_file(psg_file)
        seg += psg.to_numpy()

    # set lower jaw voxels to 2
    psg_file = psg_dir / 'LOWER_JAW' / 'LOWER_JAW.psg'
    psg = PSG.read_file(psg_file)
    seg += psg.to_numpy()

    try:
        write_nifti(dicom.to_numpy(), dir_path / 'image.nii.gz', dicom.affine)
        write_nifti(seg, dir_path / 'seg.nii.gz', dicom.affine)
    except OSError:
        pass

    return dir_path


def remove_nifti_files(root: Path) -> None:
    for path in root.glob('**/*.nii.gz'):
        os.remove(path)


def write_nifti_files(root: Path) -> None:    
    dirs = sorted([p for p in root.glob('*') if p.is_dir()])
    dir_iter = tqdm(p.imap(convert_case, dirs), total=len(dirs))
    for dir_path in dir_iter:
        dir_iter.set_description(f'Processed {dir_path.stem}')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    p = Pool(cpu_count())
    root = Path('/mnt/diag/nielsvannistelrooij/Fabian_split')

    remove_nifti_files(root)
    write_nifti_files(root)
