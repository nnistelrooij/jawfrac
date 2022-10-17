import nibabel
import numpy as np
import pytorch_lightning as pl
import yaml

from jawfrac.datamodules import JawFracDataModule
from jawfrac.models import LinearDisplacedJawFracModule


def infer():
    with open('jawfrac/config/jawfrac_linear_displaced.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    dm = JawFracDataModule(
        linear=True,
        displacements=True,
        seed=config['seed'],
        **config['datamodule'],
    )

    model = LinearDisplacedJawFracModule.load_from_checkpoint(
        'checkpoints/new_fractures_linear_displaced_patch_size=96.ckpt',
        num_classes=dm.num_classes,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.test(model, datamodule=dm)

    for (file, _), pred in zip(dm.predict_dataset.files, preds):
        pred = pred.cpu().numpy().astype(np.uint16)

        img = nibabel.load(dm.root / file)
        affine = img.affine

        file = file.parent / 'frac_pred2.nii.gz'
        img = nibabel.Nifti1Image(pred, affine)
        nibabel.save(img, dm.root / file)


if __name__ == '__main__':
    infer()
