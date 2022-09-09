import pytorch_lightning as pl
import yaml
from tqdm import tqdm

from mandibles.datamodules import MandibleSemSegDataModule
from mandibles.models import VoxelClassifier
import nibabel
import numpy as np


def infer():
    with open('mandibles/config/semseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['batch_size'] = 1
    dm = MandibleSemSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = VoxelClassifier.load_from_checkpoint(
        'checkpoints/mandible4.ckpt',
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)

    for i, volume in enumerate(tqdm(preds, desc='Writing NIfTI files')):
        # get original scan
        path = dm.root / dm.pred_dataset.files[i]
        img = nibabel.load(path)
        affine = img.affine

        # save to storage
        img = nibabel.Nifti1Image(volume.astype(np.uint16), affine)
        nibabel.save(img, path.parent / 'mandible.nii.gz')


if __name__ == '__main__':
    infer()
