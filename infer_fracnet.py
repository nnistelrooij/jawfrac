from time import perf_counter

import nibabel
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import yaml

from jawfrac.datamodules import FracNetDataModule
from jawfrac.models import FracNet


def infer(regex_filter: str=''):
    with open('jawfrac/config/fracnet.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if regex_filter:
            config['datamodule']['regex_filter'] = regex_filter

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].pop('batch_size')
    dm = FracNetDataModule(
        linear=True,
        displacements=True,
        seed=config['seed'],
        batch_size=1,
        **config['datamodule'],
    )

    model = FracNet.load_from_checkpoint(
        'checkpoints/fracnet.ckpt',
        num_classes=dm.num_classes,
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

        img = nibabel.load(path)
        file = path.parent / 'frac_pred_fracnet.nii.gz'
        img = nibabel.Nifti1Image(pred.cpu().numpy().astype(np.uint16), img.affine)
        nibabel.save(img, file)


if __name__ == '__main__':
    for regex_filter in [
        '/'
    ]:
        t = perf_counter()

        infer(regex_filter=f'Annotation UK/{regex_filter}/')
        print(f'Time: {perf_counter() - t}.')
