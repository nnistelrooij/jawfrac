import pytorch_lightning as pl
import yaml

from fractures.datamodules import JawFracPatchDataModule
from fractures.models import PatchROI


def infer():
    with open('fractures/config/patchseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['batch_size'] = 1
    dm = JawFracPatchDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = PatchROI.load_from_checkpoint(
        'checkpoints/fracture2.ckpt',
        in_channels=dm.num_channels,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    dm.setup('fit')
    trainer.validate(model, datamodule=dm)


if __name__ == '__main__':
    infer()
