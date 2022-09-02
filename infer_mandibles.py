import pytorch_lightning as pl
import yaml
from tqdm import tqdm

from mandibles.datamodules import MandibleSemSegDataModule
from mandibles.models import VoxelClassifier
import nibabel
import numpy as np
from scipy import ndimage


def infer():
    with open('mandibles/config/semseg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['batch_size'] = 1
    dm = MandibleSemSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )

    model = VoxelClassifier.load_from_checkpoint(
        'checkpoints/mandible.ckpt',
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        **config['model'],
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config['model']['epochs'],
    )
    preds = trainer.predict(model, datamodule=dm)

    for i, volume in enumerate(preds):
        # get original scan
        path = dm.root / dm.pred_dataset.files[i]
        print(path)
        img = nibabel.load(path)
        img_data = np.asarray(img.dataobj)
        affine = img.affine

        # label, _ = ndimage.label(
        #     input=img_data >= 700,
        #     structure=ndimage.generate_binary_structure(3, 2),
        # )
        # volume[label == 1] = True

        # dilate sparse predictions
        seg = volume.cpu().numpy()
        seg = ndimage.binary_dilation(
            input=seg,
            structure=ndimage.generate_binary_structure(3, 2),
            iterations=5,
            mask=np.asarray(img.dataobj) >= 300,
        )
        seg = seg.astype(np.uint16)

        # save to storage
        img = nibabel.Nifti1Image(seg, affine)
        nibabel.save(img, path.parent / 'mandible.nii.gz')


if __name__ == '__main__':
    preds = infer()
