import pytorch_lightning as pl
import yaml
from tqdm import tqdm

from mandibles.datamodules import MandiblePatchSegDataModule
from mandibles.models import MandiblePatchSegModule
import nibabel
import numpy as np


def infer():
    with open('mandibles/config/patchseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['datamodule']['batch_size'] = 1
    dm = MandiblePatchSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = MandiblePatchSegModule.load_from_checkpoint(
        'checkpoints/mandibles.ckpt',
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
        path = dm.root / dm.predict_dataset.files[i]
        img = nibabel.load(path)
        affine = img.affine

        # save to storage
        volume = volume.cpu().numpy().astype(np.uint16)
        img = nibabel.Nifti1Image(volume, affine)
        nibabel.save(img, path.parent / 'mandible.nii.gz')


if __name__ == '__main__':
    infer()
