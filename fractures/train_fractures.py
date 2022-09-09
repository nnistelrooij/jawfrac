import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from fractures.datamodules import JawFracPatchDataModule
from fractures.models import PatchROI


def train():
    with open('fractures/config/patchseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    dm = JawFracPatchDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = PatchROI(in_channels=dm.num_channels, **config['model'])


    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=config['version'],
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)



    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='epoch',
        mode='max',
        filename='weights-{epoch:02d}',
    )
    loss_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='loss/val',
        filename='weights-{epoch:02d}',
    )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
        logger=logger,
        # accumulate_grad_batches=4,
        gradient_clip_val=35,
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    train()
