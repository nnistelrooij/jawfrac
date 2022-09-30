from typing import List, Optional

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.convnet import ConvTransposeBlock
from jawfrac.nn.modules.mandibles import MandibleNet
from jawfrac.nn.modules.swin_unetr import SwinUNETRBackbone
from jawfrac.nn.modules.unet import Decoder, Encoder


class JawFracNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
        backbone: str,
        coords: Optional[str],
        checkpoint_path: str='',
    ) -> None:
        super().__init__()

        assert coords != 'sparse' or backbone == 'conv', (
            'Cannot combine swin backbone with sparse coordinates.'
        )

        if backbone == 'conv':
            self.encoder = Encoder(
                in_channels=2 + 3 * (coords == 'dense'),
                channels_list=channels_list,
            )
            self.decoder = Decoder(
                num_classes=1,
                channels_list=[128 + 3 * (coords == 'sparse'), 64, 32, 16],
            )
        elif backbone == 'swin':
            self.unet = SwinUNETRBackbone(
                img_size=64,
                in_channels=2 + 3 * (coords == 'dense'),
                out_channels=1,
            )
        else:
            raise ValueError(f'Backbone not recognized: {backbone}.')

        if coords == 'sparse':
            self.init_sparse_coords()

        self.head = nn.Conv3d(
            in_channels=32 - 8 * (backbone == 'swin'),
            out_channels=1,
            kernel_size=1,
        )

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = {k: v for k, v in state_dict.items() if 'man' not in k}
            state_dict = {k[9:]: v for k, v in state_dict.items()}
            self.load_state_dict(state_dict)
            self.requires_grad_(False)

        self.backbone = backbone
        self.coords = coords

    def init_sparse_coords(self):
        self.coords_linear = nn.Sequential(
            nn.Linear(3, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(),
        )
        self.coords_conv = nn.Sequential(
            ConvTransposeBlock(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
            ),
            ConvTransposeBlock(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
            ),
        )

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
        coords: TensorType['B', 3, torch.float32],
        mandible: TensorType['B', 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'D', 'H', 'W', torch.float32]:
        x = torch.cat((x, mandible.unsqueeze(dim=1)), dim=1)

        if self.coords == 'dense':
            voxel_coords = torch.linspace(-0.4, 0.4, steps=x.shape[-1])
            voxel_coords = torch.cartesian_prod(*voxel_coords.tile(3, 1))
            voxel_coords = voxel_coords.reshape(1, *x.shape[2:], 3).to(coords)
            voxel_coords = voxel_coords + coords.reshape(-1, 1, 1, 1, 3)
            voxel_coords[..., 0] = torch.abs(voxel_coords[..., 0])  # left-right symmetry
            voxel_coords = voxel_coords.permute(0, 4, 1, 2, 3)
            x = torch.cat((x, voxel_coords), dim=1)

        if self.backbone == 'conv':
            xs = self.encoder(x)

        if coords == 'sparse':
            coords = self.coords_linear(coords)
            coords = coords.reshape(-1, 3, 2, 2, 2)
            coords = self.coords_conv(coords)
            xs[0] = torch.cat((xs[0], coords), dim=1)

        if self.backbone == 'conv':
            x = self.decoder(xs)
        elif self.backbone == 'swin':
            _, x = self.unet(x)

        x = self.head(x)

        return x.squeeze(dim=1)
