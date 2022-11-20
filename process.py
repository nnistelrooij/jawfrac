from pathlib import Path

import nibabel
import numpy as np
import pytorch_lightning as pl
import yaml

from jawfrac.datamodules import JawFracDataModule, MandibleSegDataModule
from jawfrac.models import MandibleSegModule, LinearDisplacedJawFracModule
from jawfrac.models.common import fill_source_volume



def infer_mandible():
    with open('jawfrac/config/mandibles.yaml') as f:
        config = yaml.safe_load(f)
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

    return shape


def infer_fractures(shape):
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
    (out_dir / 'main.nii.gz').unlink()
    (out_dir / 'mandible_pred.nii.gz').rename(out_dir / 'mandible.nii.gz')


if __name__ == '__main__':
    shape = infer_mandible()
    infer_fractures(shape)
