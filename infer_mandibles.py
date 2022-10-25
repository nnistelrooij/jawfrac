import pytorch_lightning as pl
import yaml
import torch.nn as nn
from tqdm import tqdm

from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule
import nibabel
import numpy as np


def infer():
    with open('jawfrac/config/mandibles.yaml', 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['datamodule'].pop('batch_size')
    dm = MandibleSegDataModule(
        seed=config['seed'], batch_size=1, **config['datamodule'],
    )

    model = MandibleSegModule.load_from_checkpoint(
        'checkpoints/mandibles.ckpt',
        num_classes=dm.num_classes,
        batch_size=batch_size,
        **config['model'],
    )
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.track_running_stats = False

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
        nibabel.save(img, path.parent / 'mandible200.nii.gz')


if __name__ == '__main__':
    while True:
        infer()
