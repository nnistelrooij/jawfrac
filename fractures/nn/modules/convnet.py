from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class ConvNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        channels_list: List[int],
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(ConvBlock(
            in_channels, channels_list[0],
        ))
        for i in range(len(channels_list) - 1):
            self.layers.append(nn.MaxPool3d(kernel_size=2))
            self.layers.append(ConvBlock(
                channels_list[i], channels_list[i + 1],
            ))

        self.head = nn.Linear(channels_list[-1], 1)

    def forward(
        self,
        x: TensorType['B', 'C', 'H', 'W', 'D', torch.float32],        
    ) -> TensorType['B', torch.float32]:
        x = self.layers(x)
        x = x.mean(dim=(2, 3, 4))
        x = self.head(x)

        return x


class FracNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels,
            [16, 32, 64, 128],
        )



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
        x = self.layers[0](x)

        xs = [x]
        for layer in self.layers[1:]:
            x = F.max_pool3d(x, kernel_size=2)
            x = layer(x)
            xs.append(x)

        return xs



class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        padding=1,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        x: TensorType['B', 'C', 'H', 'W', 'D', torch.float32],
    ) -> TensorType['B', 'C', 'H', 'W', 'D', torch.float32]:
        return self.layers(x)

class ConvTransposeBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        x: TensorType['B', 'C', 'H', 'W', 'D', torch.float32],
    ) -> TensorType['B', 'C', 'H', 'W', 'D', torch.float32]:
        return self.layers(x)
