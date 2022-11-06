import nibabel
import numpy as np
import pytorch_lightning as pl
import yaml

import jawfrac.data.transforms as T
from jawfrac.datamodules import JawFracDataModule, MandibleSegDataModule
from jawfrac.models import MandibleSegModule, LinearDisplacedJawFracModule



def infer_mandible():
    with open('jawfrac/config/mandibles.yaml') as f:
        config = yaml.safe_load(f)

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
    mask = trainer.predict(model, datamodule=dm)[0]
    mask = mask.cpu().numpy().astype(np.uint16)

    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file)

    img = nibabel.Nifti1Image(mask, img.affine)
    nibabel.save(img, file.parent / 'mandible200.nii.gz')


def infer_fractures():
    with open('jawfrac/config/jawfrac_linear_displaced.yaml') as f:
        config = yaml.safe_load(f)

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
    mask = trainer.predict(model, datamodule=dm)[0]
    mask = mask.cpu().numpy().astype(np.uint16)

    file = dm.root / dm.predict_dataset.files[0][0]
    img = nibabel.load(file)

    img = nibabel.Nifti1Image(mask, img.affine)
    nibabel.save(img, file.parent / 'fractures.nii.gz')


if __name__ == '__main__':
    infer_mandible()
    infer_fractures()
