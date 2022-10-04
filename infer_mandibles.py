import pytorch_lightning as pl
import yaml
import torch
from tqdm import tqdm

from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule
import nibabel
import numpy as np


def infer():
    with open('jawfrac/config/mandibles.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['datamodule']['batch_size'] = 1
    dm = MandibleSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = MandibleSegModule.load_from_checkpoint(
        'checkpoints/mandibles_positions_size=0.4.ckpt',
        num_classes=dm.num_classes,
        **config['model'],
    )
    torch.save(model.model.unet.state_dict(), 'checkpoints/unet.ckpt')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)

    for i, volume in enumerate(tqdm(preds, desc='Writing NIfTI files')):
        # get original scan
        path = dm.root / dm.predict_dataset.files[i][0]
        img = nibabel.load(path)
        affine = img.affine

        # save to storage
        volume = volume.cpu().numpy().astype(np.uint16)
        img = nibabel.Nifti1Image(volume, affine)
        nibabel.save(img, path.parent / 'mandible2.nii.gz')


if __name__ == '__main__':
    while True:
        infer()
