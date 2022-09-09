from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from fractures.nn.modules.convnet import (
    ConvBlock,
    ConvTransposeBlock,
)


class FracNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels,
            [16, 32, 64],
        )

        self.decoder = Decoder(
            num_classes,
            [64, 128, 64, 32, 16],
        )

    def forward(
        self,
        x: TensorType['B', 'C', 'H', 'W', 'D', torch.float32],
    ) -> TensorType['B', 'C', 'H', 'W', 'D', torch.float32]:
        xs = self.encoder(x)
        x = self.decoder(xs)

        return x


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
        for layer in self.layers:
            x = layer(x)
            xs = [x] + xs
            x = F.max_pool3d(x, kernel_size=2)

        return [x] + xs


class Decoder(nn.Module):

    def __init__(
        self,
        num_classes: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(channels_list) - 2):
            self.layers.append(nn.Sequential(
                ConvBlock(channels_list[i], channels_list[i + 1]),
                ConvTransposeBlock(channels_list[i + 1], channels_list[i + 2], 3),
            ))

        self.head = nn.Conv3d(channels_list[-2], num_classes, 3, padding=1)

    def forward(
        self,
        xs: List[TensorType['B', 'C', 'H', 'W', 'D', torch.float32]],
    ) -> TensorType['B', 'C', 'H', 'W', 'D', torch.float32]:
        x = xs[0]

        for layer, skip_x in zip(self.layers, xs[1:]):
            x = layer(x)
            x = torch.cat((x, skip_x), dim=1)

        x = self.head(x)

        return x.squeeze(1)


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float,
        gamma: float,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        x: TensorType['B', 'D', 'H', 'W', torch.float32],
        y: TensorType['B', 'D', 'H', 'W', torch.int64],
    ) -> TensorType[torch.float32]:
        bce_loss = self.bce(x, y.float())

        probs = torch.sigmoid(x)
        pt = y * probs + (1 - y) * (1 - probs)
        alphat = y * self.alpha + (1 - y) * (1 - self.alpha)
        bce_loss *= alphat * (1 - pt) ** self.gamma

        return bce_loss.mean()
