from functools import partial

import nibabel
import numpy as np
import pytorch_lightning as pl
import torch
from torchtyping import TensorType
import yaml

import jawfrac.data.transforms as T
from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule, LinearDisplacedJawFracModule



def infer_mandible():
    with open('jawfrac/config/mandibles.yaml') as f:
        config = yaml.safe_load(f)
        config['datamodule']['num_workers'] = 0

    pl.seed_everything(config['seed'])

    dm = MandibleSegDataModule(seed=config['seed'], **config['datamodule'])
    dm.setup('predict')
    batch = next(iter(dm.predict_dataloader()))
    
    model = MandibleSegModule.load_from_checkpoint(
        'checkpoints/mandibles.ckpt',
        num_classes=dm.num_classes,
        return_source_volume=False,
        **config['model'],
    )
    mask = model.predict_step(batch)

    return dm, mask


def infer_fractures(
    dm: pl.LightningDataModule,
    mandible: TensorType['D', 'H', 'W', torch.bool],
):
    with open('jawfrac/config/fractures_linear_displaced.yaml') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'])

    dm.predict_dataset.transform = T.Compose(
        partial(dict, mandible=mandible),
        T.MandibleCrop(**config['datamodule']['mandible_crop']),
        T.PatchIndices(
            patch_size=config['datamdule']['patch_size'],
            stride=config['datamdule']['stride'],
        ),
        T.BonePatchIndices(),
        dm.default_transforms,
    )
    batch = next(iter(dm.predict_dataloader()))
    
    model = LinearDisplacedJawFracModule.load_from_checkpoint(
        'checkpoints/fractures_linear_displaced.ckpt',
        num_classes=len(config['datamodule']['class_label_to_idx']),
        **config['model'],
    )
    mask = model.predict_step(batch)

    return mask


def process():
    dm, mandible = infer_mandible()
    fractures = infer_fractures(dm)

    out = np.zeros(mandible.shape, dtype=np.int16)
    out[mandible.cpu()] = 1
    out[fractures.cpu()] = 2

    file = dm.predict_dataset.files[0][0]

    img = nibabel.load(file)
    img = nibabel.Nifti1Image(out, img.affine)
    nibabel.save(img, '/output/jawfrac.nii.gz')


if __name__ == '__main__':
    process()
