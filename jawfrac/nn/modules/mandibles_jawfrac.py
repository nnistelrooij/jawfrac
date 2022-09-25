from typing import List, Optional

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.convnet import ConvTransposeBlock
from jawfrac.nn.modules.mandibles import MandibleNet
from jawfrac.nn.modules.unet import Decoder, Encoder


class MandibleFracNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_awms: int,
        num_classes: int,
        channels_list: List[int],
        checkpoint_path: str,
        backbone: str,
        coords: Optional[str],
    ) -> None:
        super().__init__()

        self.mandible_net = MandibleNet(
            num_awms=num_awms,
            num_classes=num_classes,
            channels_list=channels_list,
            checkpoint_path=checkpoint_path,
            backbone=backbone,
        )
        self.mandible_net.requires_grad_(False)

        self.encoder = Encoder(
            in_channels=2 + 3 * (coords == 'dense'),
            channels_list=channels_list,
        )

        if coords == 'sparse':
            self.init_sparse_coords()

        self.decoder = Decoder(
            num_classes=1,
            channels_list=[128 + 3 * (coords == 'sparse'), 64, 32, 16],
        )

        self.head = nn.Conv3d(
            in_channels=channels_list[1],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

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
    ) -> TensorType['B', 'D', 'H', 'W', torch.float32]:
        coords, mandible = self.mandible_net(x)

        x = torch.cat((x, mandible.unsqueeze(dim=1)), dim=1)

        if self.coords == 'dense':
            voxel_coords = torch.linspace(-0.4, 0.4, steps=x.shape[-1])
            voxel_coords = torch.cartesian_prod(*voxel_coords.tile(3, 1))
            voxel_coords = voxel_coords.reshape(1, *x.shape[2:], 3).to(coords)
            voxel_coords = voxel_coords + coords.reshape(-1, 1, 1, 1, 3)
            voxel_coords[..., 0] = torch.abs(voxel_coords[..., 0])  # left-right symmetry
            voxel_coords = voxel_coords.permute(0, 4, 1, 2, 3)
            x = torch.cat((x, voxel_coords), dim=1)

        xs = self.encoder(x)

        if coords == 'sparse':
            coords = self.coords_linear(coords)
            coords = coords.reshape(-1, 3, 2, 2, 2)
            coords = self.coords_conv(coords)
            xs[0] = torch.cat((xs[0], coords), dim=1)

        x = self.decoder(xs)
        x = self.head(x)

        return x.squeeze(dim=1)
