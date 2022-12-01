from pathlib import Path
import shutil
from time import perf_counter

import nibabel
import numpy as np
import pytorch_lightning as pl
import torch
from torchtyping import TensorType
import yaml

from jawfrac.datamodules import JawFracDataModule, MandibleSegDataModule
from jawfrac.models import MandibleSegModule, LinearDisplacedJawFracModule
from jawfrac.models.common import fill_source_volume



def infer_mandible(regex_filter: str='') -> TensorType[3, torch.int64]:
    with open('jawfrac/config/mandibles.yaml') as f:
        config = yaml.safe_load(f)
        if regex_filter:
            config['datamodule']['regex_filter'] = regex_filter
        out_dir = Path(config['work_dir'])

    pl.seed_everything(config['seed'])

    dm = MandibleSegDataModule(seed=config['seed'], **config['datamodule'])
    
    model = MandibleSegModule.load_from_checkpoint(
        'checkpoints/mandibles.ckpt',
        num_classes=dm.num_classes,
        batch_size=config['datamodule']['batch_size'],
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)[0]
    intensities, mandible, out, affine, shape = preds


    # save mandible segmentation of original volume to storage
    out = out.cpu().numpy().astype(np.uint16)

    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file)

    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, out_dir / 'mandible_pred.nii.gz')


    # save pre-processed intensities and mandible to storage
    intensities = intensities.cpu().numpy().astype(np.int16)
    mandible = mandible.cpu().numpy().astype(np.uint16)

    intensities = nibabel.Nifti1Image(intensities, affine)
    mandible = nibabel.Nifti1Image(mandible, affine)

    nibabel.save(intensities, out_dir / 'main.nii.gz')
    nibabel.save(mandible, out_dir / 'mandible.nii.gz')


    # copy original file to output folder
    shutil.copy(file, out_dir / 'main_source.nii.gz')


    return shape


def infer_fractures(shape: TensorType[3, torch.int64]):
    with open('jawfrac/config/jawfrac_linear_displaced.yaml') as f:
        config = yaml.safe_load(f)
        out_dir = Path(config['work_dir'])

    pl.seed_everything(config['seed'])

    dm = JawFracDataModule(
        seed=config['seed'],
        linear=True,
        displacements=True,
        **config['datamodule'],
    )
    
    model = LinearDisplacedJawFracModule.load_from_checkpoint(
        'checkpoints/fractures.ckpt',
        num_classes=dm.num_classes,
        batch_size=config['datamodule']['batch_size'],
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)[0]
    mask, _, affine, _ = preds

    out = fill_source_volume(mask, affine, shape)


    # save fractures segmentation of original volume to storage
    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file)

    out = out.cpu().numpy().astype(np.uint16)
    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, out_dir / 'fractures.nii.gz')


    # remove pre-processed volumes
    (out_dir / 'main_source.nii.gz').rename(out_dir / 'main.nii.gz')
    (out_dir / 'mandible_pred.nii.gz').rename(out_dir / 'mandible.nii.gz')


if __name__ == '__main__':
    mandible_times, jawfracnet_times = [], []
    for regex_filter in [
        #'Annotation UK/1/', 'Annotation UK/109/', 'Annotation UK/11/', 'Annotation UK/114/', 'Annotation UK/119/', 'Annotation UK/121/', 'Annotation UK/122/', 'Annotation UK/125/', 'Annotation UK/126/', 'Annotation UK/127/', 'Annotation UK/132/', 'Annotation UK/134/', 'Annotation UK/141/', 'Annotation UK/149/', 'Annotation UK/150/', 'Annotation UK/157/', 'Annotation UK/159/', 'Annotation UK/17/', 'Annotation UK/173/', 'Annotation UK/182/', 'Annotation UK/186/', 'Annotation UK/188/', 'Annotation UK/189/', 'Annotation UK/192/', 'Annotation UK/194/', 'Annotation UK/25/', 'Annotation UK/31/', 'Annotation UK/34/', 'Annotation UK/35/', 'Annotation UK/36/', 'Annotation UK/42/', 'Annotation UK/45/', 'Annotation UK/48/', 'Annotation UK/55/', 'Annotation UK/67/',
        #'Controls/Patient 56-60/DICOM/Patient 59/', 'Controls/Patient 61-65/DICOM/Patient 64/', 'Controls/Patient 28-40/DICOM/Patient 36/', 'Controls/Patient 51-55/DICOM/Patient 51/', 'Controls/Patient 41-45/DICOM/Patient 42/', 'Controls/Patient 4/', 'Controls/Patient 76-80/DICOM/Patient 76/', 'Controls/Patient 56-60/DICOM/Patient 60/', 'Controls/Patient 46-50/DICOM/Patient 48/', 'Controls/Patient 16-21/DICOM/Patient 19/', 'Controls/Patient 5-10/DICOM/Patient 9/', 'Controls/Patient 66-70/DICOM/Patient 66/', 'Controls/Patient 5-10/DICOM/Patient 7/', 'Controls/Patient 71-75/DICOM/Patient 71/', 'Controls/Patient 46-50/DICOM/Patient 47/', 'Controls/Patient 76-80/DICOM/Patient 77/', 'Controls/Patient 41-45/DICOM/Patient 41/', 'Controls/Patient 56-60/DICOM/Patient 57/', 'Controls/Patient 66-70/DICOM/Patient 67/', 'Controls/Patient 61-65/DICOM/Patient 63/', 'Controls/Patient 16-21/DICOM/Patient 17/', 'Controls/Patient 56-60/DICOM/Patient 58/', 'Controls/Patient 76-80/DICOM/Patient 80/', 'Controls/Patient 11-15/DICOM/Patient 15/', 'Controls/Patient 61-65/DICOM/Patient 65/', 'Controls/Patient 66-70/DICOM/Patient 69/', 'Controls/Patient 11-15/DICOM/Patient 13/', 'Controls/Patient 81-85/DICOM/Patient 84/', 'Controls/Patient 51-55/DICOM/Patient 53/', 'Controls/Patient 23-27/DICOM/Patient 23/', 'Controls/Patient 41-45/DICOM/Patient 44/', 'Controls/Patient 11-15/DICOM/Patient 14/', 'Controls/Patient 66-70/DICOM/Patient 70/', 'Controls/Patient 22/DICOM/STD00001/', 'Controls/Patient 41-45/DICOM/Patient 45/',
        'Annotation UK/17/',
    ]:
        t = perf_counter()

        shape = infer_mandible(regex_filter=regex_filter)
        mandible_times.append(perf_counter() - t)
        print(f'Time for mandible segmentation: {perf_counter() - t}.')
        
        infer_fractures(shape)
        jawfracnet_times.append(perf_counter() - t)
        print(f'Time for JawFracNet: {perf_counter() - t}.')

    mandible_times = np.array(mandible_times)
    jawfracnet_times = np.array(jawfracnet_times)

    print(f'Total time for mandible: {mandible_times.mean()} +- {mandible_times.std()}.')
    print(f'Total time for JawFracNet: {jawfracnet_times.mean()} +- {jawfracnet_times.std()}.')
