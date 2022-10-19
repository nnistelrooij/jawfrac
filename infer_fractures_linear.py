from typing import Any, Dict

import nibabel
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from jawfrac.datamodules import JawFracDataModule
from jawfrac.models import LinearJawFracModule


def fix_batchnorm(
    config: Dict[str, Any],
    model: torch.nn.Module,
) -> None:
    mandible_ckpt_path = config['model']['first_stage']['checkpoint_path']
    mandible_state_dict = torch.load(mandible_ckpt_path)['state_dict']

    model_state_dict = model.state_dict()
    for key in model_state_dict.copy():
        if 'running' not in key:
            continue

        if 'mandible' in key:
            old_key = key.replace('mandible_net', 'model')
            model_state_dict[key] = mandible_state_dict[old_key]

    model.load_state_dict(model_state_dict)


def infer():
    with open('jawfrac/config/jawfrac_linear.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    dm = JawFracDataModule(
        linear=True,
        displacements=False,
        seed=config['seed'],
        **config['datamodule'],
    )

    model = LinearJawFracModule.load_from_checkpoint(
        'checkpoints/fractures_linear.ckpt',
        num_classes=dm.num_classes,
        **config['model'],
    )
    fix_batchnorm(config, model)

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

        file = file.parent / 'frac_pred.nii.gz'
        img = nibabel.Nifti1Image(pred, affine)
        nibabel.save(img, dm.root / file)


if __name__ == '__main__':
    infer()
