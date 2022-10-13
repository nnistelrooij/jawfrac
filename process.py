import nibabel
import numpy as np
import pytorch_lightning as pl
import yaml

from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule, LinearDisplacedJawFracModule



def infer_mandible():
    with open('jawfrac/config/mandibles.yaml') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'])

    dm = MandibleSegDataModule(seed=config['seed'], **config['datamodule'])
    dm.setup('predict')
    batch = next(iter(dm.predict_dataloader()))
    
    model = MandibleSegModule.load_from_checkpoint(
        'checkpoints/mandible.ckpt',
        num_classes=dm.num_classes,
        return_source_volume=False,
        **config['model'],
    )
    mask = model.predict_step(batch)

    return dm, mask


def infer_fractures(dm, mandible):
    with open('jawfrac/config/fractures_linear_displaced.yaml') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'])

    batch = next(iter(dm.predict_dataloader()))
    batch = batch[:1] + mask + batch[1:]
    
    model = LinearDisplacedJawFracModule.load_from_checkpoint(
        'checkpoints/fractures_linear_displaced',
        num_classes=len(config['datamodule']['class_label_to_idx']),
        **config['model'],
    )
    mask = model.predict_step(batch)

    return mask


def process():
    dm, mandible = infer_mandible()
    fractures = infer_fractures(dm, mandible)
    fractures = fractures.cpu().numpy().astype(np.int16)

    file = dm.predict_dataset.files[0][0]

    img = nibabel.load(file)
    img = nibabel.Nifti1Image(fractures, img.affine)
    nibabel.save(img, '/output/frac.nii.gz')


if __name__ == '__main__':
    process()
