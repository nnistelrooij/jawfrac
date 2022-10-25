import nibabel
import numpy as np
import pytorch_lightning as pl
import yaml
from tqdm import tqdm

from jawfrac.datamodules import JawFracDataModule
from jawfrac.models import LinearDisplacedJawFracModule


def infer():
    with open('jawfrac/config/jawfrac_linear_displaced.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    batch_size = config['datamodule'].pop('batch_size')
    dm = JawFracDataModule(
        linear=True,
        displacements=True,
        seed=config['seed'],
        batch_size=1,
        **config['datamodule'],
    )

    model = LinearDisplacedJawFracModule.load_from_checkpoint(
        'checkpoints/old_fractures_linear_displaced_patch_size=64.ckpt',
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

    for i, pred in enumerate(tqdm(preds)):
        path = dm.root / dm.predict_dataset.files[i][0]

        label = nibabel.load(path.parent / 'label.nii.gz')
        label = np.asarray(label.dataobj)
        label[(label == 0) & pred.cpu().numpy()] = 3

        img = nibabel.load(path)
        file = path.parent / 'frac_pred.nii.gz'
        img = nibabel.Nifti1Image(pred.cpu().numpy().astype(np.uint16), img.affine)
        nibabel.save(img, file)


if __name__ == '__main__':
    while True:
        infer()
