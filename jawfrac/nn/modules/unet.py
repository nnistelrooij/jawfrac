from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from jawfrac.nn.modules.convnet import (
    ConvBlock,
    ConvTransposeBlock,
)
from jawfrac.nn.modules.gapm import GrayscaleAdaptivePerceptionModule


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels_list: List[int],
        num_awms: int,
        checkpoint_path: str='',
    ) -> None:
        super().__init__()

        self.gapm = GrayscaleAdaptivePerceptionModule(
            num_awms=num_awms,
        )

        self.encoder = Encoder(
            1 + num_awms,
            channels_list,
        )

        self.decoder = Decoder(
            num_classes,
            channels_list[::-1],
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint)

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
        TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ]:
        x = self.gapm(x)
        xs = self.encoder(x)
        x = self.decoder(xs)

        return xs[0], x


class Encoder(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        channels_list = [in_channels] + channels_list
        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.layers.append(
                ConvBlock(channels_list[i], channels_list[i + 1], 3, 1)
            )

    def forward(
        self,
        x: TensorType['B', 'C', 'H', 'W', 'D', torch.float32],
    ) -> List[TensorType['B', 'C', 'H', 'W', 'D', torch.float32]]:
        xs = []
        for layer in self.layers[:-1]:
            x = layer(x)
            xs = [x] + xs
            x = F.max_pool3d(x, kernel_size=2)

        x = self.layers[-1](x)
        xs = [x] + xs

        return xs


class Decoder(nn.Module):

    def __init__(
        self,
        num_classes: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.layers.append(nn.Sequential(
                ConvBlock(2 * channels_list[i], channels_list[i]) if i > 0 else nn.Identity(),
                ConvTransposeBlock(channels_list[i], channels_list[i + 1], 3),
            ))

    def forward(
        self,
        xs: List[TensorType['B', 'C', 'H', 'W', 'D', torch.float32]],
    ) -> TensorType['B', 'C', 'H', 'W', 'D', torch.float32]:
        x = xs[0]

        for layer, skip_x in zip(self.layers, xs[1:]):
            x = layer(x)
            x = torch.cat((x, skip_x), dim=1)
            
        return x
