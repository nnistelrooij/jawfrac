from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.convnet import ConvTransposeBlock
from jawfrac.nn.modules.gapm import GrayscaleAdaptivePerceptionModule
from jawfrac.nn.modules.swin_unetr import SwinUNETRBackbone
from jawfrac.nn.modules.unet import Decoder, Encoder


class JawFracNet(nn.Module):

    def __init__(
        self,
        num_awms: int,
        num_classes: int,
        mandible_channels: int,
        channels_list: List[int],
        backbone: str,
        coords: Optional[str],
        head_kernel_size: int,
        cascade: bool,
        gapm_level: float=1036.0,
        gapm_width: float=4120.0,
        return_features: bool=False,
        checkpoint_path: str='',
    ) -> None:
        super().__init__()

        assert backbone == 'conv' or 'sparse' not in coords, (
            'Cannot combine swin backbone with sparse coordinates.'
        )

        self.gapm = GrayscaleAdaptivePerceptionModule(
            num_awms, init_level=gapm_level, init_width=gapm_width,
        )
        
        in_channels = 1 + num_awms + mandible_channels + 3 * ('dense' in coords)

        if backbone == 'conv':
            self.encoder1 = Encoder(
                in_channels=in_channels,
                channels_list=channels_list,
            )
            channels_list[-1] += 3 * ('sparse' in coords)
            self.decoder1 = Decoder(
                num_classes=1,
                channels_list=channels_list[::-1],
            )
            if cascade:
                channels_list[-1] -= 3 * ('sparse' in coords)
                self.encoder2 = Encoder(
                    in_channels=in_channels + 1,
                    channels_list=channels_list,
                )
                channels_list[-1] += 3 * ('sparse' in coords)
                self.decoder2 = Decoder(
                    num_classes=1,
                    channels_list=channels_list[::-1],
                )
        elif backbone == 'swin':
            self.unet1 = SwinUNETRBackbone(
                img_size=64,
                in_channels=in_channels,
                out_channels=1,
            )
            if cascade:
                self.unet2 = SwinUNETRBackbone(
                    img_size=64,
                    in_channels=in_channels + 1,
                    out_channels=1,
                )
        else:
            raise ValueError(f'Backbone not recognized: {backbone}.')

        if 'dynamic' in coords and 'sparse' in coords:
            self.init_sparse_coords()

        latent_features = channels_list[1] if backbone == 'conv' else 24
        self.head1 = nn.Conv3d(
            in_channels=latent_features,
            out_channels=1,
            kernel_size=head_kernel_size,
            padding=head_kernel_size // 2,
        )
        if cascade:
            self.head2 = nn.Conv3d(
                in_channels=latent_features,
                out_channels=1,
                kernel_size=head_kernel_size,
                padding=head_kernel_size // 2,
            )

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = {k: v for k, v in state_dict.items() if 'man' not in k}
            state_dict = {k[9:]: v for k, v in state_dict.items()}
            self.load_state_dict(state_dict)
            self.requires_grad_(False)

        self.backbone = backbone
        self.coords = coords
        self.cascade = cascade
        self.return_features = return_features
        self.out_channels = 1 + return_features * latent_features

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
        mandible: TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
    ) -> Union[
        TensorType['B', '[C]', 'D', 'H', 'W', torch.float32],
        Tuple[
            TensorType['B', 'D', 'H', 'W', torch.float32],
            TensorType['B', 'D', 'H', 'W', torch.float32],
        ],
    ]:
        x = self.gapm(x)

        mandible = mandible.reshape(x.shape[:1] + (-1,) + x.shape[2:])
        x = torch.cat((x, mandible), dim=1)

        if 'dense' in self.coords:
            if 'dynamic' in self.coords:
                voxel_coords = torch.linspace(-0.15, 0.15, steps=x.shape[-1])
                voxel_coords = torch.cartesian_prod(*voxel_coords.tile(3, 1))
                voxel_coords = voxel_coords.reshape(*x.shape[2:], 1, 3)
                voxel_coords = voxel_coords.to(coords) + coords
                voxel_coords[..., 0] = torch.abs(voxel_coords[..., 0])  # left-right symmetry
            else:
                voxel_coords = coords.tile(*x.shape[2:], 1, 1)
            
            voxel_coords = voxel_coords.permute(3, 4, 0, 1, 2)
            x = torch.cat((x, voxel_coords), dim=1)

        if self.backbone == 'conv':
            xs = self.encoder1(x)

        if 'sparse' in self.coords:
            if 'dynamic' in self.coords:
                coords = self.coords_linear(coords)
                voxel_coords = coords.reshape(-1, 3, 2, 2, 2)
                voxel_coords = self.coords_conv(voxel_coords)
            else:
                voxel_coords = coords.tile(*xs[0].shape[2:], 1, 1)
                voxel_coords = voxel_coords.permute(3, 4, 0, 1, 2)

            xs[0] = torch.cat((xs[0], voxel_coords), dim=1)

        if self.backbone == 'conv':
            features = self.decoder1(xs)
        elif self.backbone == 'swin':
            _, features = self.unet1(x)

        masks1 = self.head1(features)

        if not self.cascade:
            if self.return_features:
                return torch.cat((masks1, features), dim=1)
            else:
                return masks1.squeeze(dim=1)

        x = torch.cat((x, masks1), dim=1)

        if self.backbone == 'conv':
            xs = self.encoder2(x)
            if 'sparse' in self.coords:
                xs[0] = torch.cat((xs[0], voxel_coords), dim=1)
            features = self.decoder2(xs)
        elif self.backbone == 'swin':
            _, features = self.unet2(x)

        masks2 = self.head2(features)

        if self.return_features:
            return torch.cat((masks2, features), dim=1)
        else:
            return masks1.squeeze(dim=1), masks2.squeeze(dim=1)
