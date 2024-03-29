from pathlib import Path
import shutil
from time import perf_counter

import nibabel
import numpy as np
import pytorch_lightning as pl
import SimpleITK as sitk
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
            config['datamodule']['root'] = config['datamodule']['root'] + regex_filter
            config['datamodule']['regex_filter'] = ''
        out_dir = Path(config['work_dir']).resolve()

    uuid = ''
    for path in Path(config['datamodule']['root']).glob('**/*'):
        print(path)
        if path.suffix != '.mha':
            continue

        img = sitk.ReadImage(path)
        sitk.WriteImage(img, out_dir / 'image.nii.gz')

        config['datamodule']['root'] = out_dir
        uuid = path.stem

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
    intensities, mandible, out, affine, shape, preprocess_time, counter = preds


    # save mandible segmentation of original volume to storage
    out = out.cpu().numpy().astype(np.uint16)
    if not np.any(out):
        return None

    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file)

    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, out_dir / 'mandible_pred.nii.gz')

    inference_time = perf_counter() - counter


    # save pre-processed intensities and mandible to storage
    intensities = intensities.cpu().numpy().astype(np.int16)
    mandible = mandible.cpu().numpy().astype(np.uint16)

    intensities = nibabel.Nifti1Image(intensities, affine)
    mandible = nibabel.Nifti1Image(mandible, affine)

    nibabel.save(intensities, out_dir / 'main.nii.gz')
    nibabel.save(mandible, out_dir / 'mandible.nii.gz')


    # copy original file to output folder
    shutil.copy(file, out_dir / 'main_source.nii.gz')


    return uuid, shape, preprocess_time, inference_time


def infer_fractures(uuid: str, shape: TensorType[3, torch.int64]):
    with open('jawfrac/config/jawfrac_linear_displaced.yaml') as f:
        config = yaml.safe_load(f)
        out_dir = Path(config['work_dir']).resolve()

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
    mask, _, affine, _, preprocess_time, counter = preds

    out = fill_source_volume(mask, affine, shape)


    # save fractures segmentation of original volume to storage
    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file.parent / 'main_source.nii.gz')

    out = out.cpu().numpy().astype(np.uint16)
    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, out_dir / 'fractures.nii.gz')

    inference_time = perf_counter() - counter


    # remove pre-processed volumes
    (out_dir / 'main_source.nii.gz').rename(out_dir / 'main.nii.gz')
    (out_dir / 'mandible_pred.nii.gz').rename(out_dir / 'mandible.nii.gz')

    # convert to .mha files
    if uuid:
        img = sitk.ReadImage(out_dir / 'mandible.nii.gz')
        mha_dir = out_dir / 'images' / 'mandible-segmentation'
        mha_dir.mkdir(parents=True)
        sitk.WriteImage(img, mha_dir / f'{uuid}.mha')

        img = sitk.ReadImage(out_dir / 'fractures.nii.gz')
        mha_dir = out_dir / 'images' / 'fractures'
        mha_dir.mkdir(parents=True)
        sitk.WriteImage(img, mha_dir / f'{uuid}.mha')

    return preprocess_time, inference_time


if __name__ == '__main__':
    preprocess_times, mandible_times, fractures_times, jawfracnet_times = [], [], [], []
    for regex_filter in ['/']:
        counter = perf_counter()

        out = infer_mandible(regex_filter=regex_filter)
        if out is None:
            print('NO MANDIBLE')
            continue
        
        uuid, shape, preprocess1_time, mandible_time = out        
        preprocess2_time, fractures_time = infer_fractures(uuid, shape)

        preprocess_times.append(preprocess1_time + preprocess2_time)
        mandible_times.append(mandible_time)
        fractures_times.append(fractures_time)
        jawfracnet_times.append(perf_counter() - counter)
        
        print(f'Time for pre-process: {preprocess_times[-1]}.')
        print(f'Time for mandible segmentation: {mandible_times[-1]}.')
        print(f'Time for fractures {fractures_times[-1]}.')
        print(f'Time for JawFracNet {jawfracnet_times[-1]}.')

    preprocess_times = np.array(preprocess_times)
    mandible_times = np.array(mandible_times)
    fractures_times = np.array(fractures_times)
    jawfracnet_times = preprocess_times + mandible_times + fractures_times

    print(f'Total time for pre-process: {preprocess_times.mean()} +- {preprocess_times.std()}.')
    print(f'Total time for mandible: {mandible_times.mean()} +- {mandible_times.std()}.')
    print(f'Total time for fractures: {fractures_times.mean()} +- {fractures_times.std()}.')
    print(f'Total time for JawFracNet: {jawfracnet_times.mean()} +- {jawfracnet_times.std()}.')
