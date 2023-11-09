import pytorch_lightning as pl
import yaml
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
        volume = volume[2].cpu().numpy().astype(np.uint8)
        img = nibabel.Nifti1Image(volume, affine)
        nibabel.save(img, path.parent / f'{path.stem[:-9]}.nii.gz')


if __name__ == '__main__':
    while True:
        infer()
