import torch
import torch.nn as nn
from torchtyping import TensorType


class AdaptiveWindowingModule(nn.Module):

    def __init__(
        self,
        out_channels: int,
        init_level: float,
        init_width: float,
    ) -> None:
        super().__init__()

        self.min = init_level - init_width / 2
        self.max = init_level + init_width / 2

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.conv2a = nn.Conv3d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
        )
        self.conv2b = nn.Conv3d(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 1, 'D', 'H', 'W', torch.float32]:
        # get features from two separate branches
        z = self.conv1(x)
        za = self.conv2a(z)
        zb = self.conv2b(z)

        # perform global average pooling
        za = za.mean(dim=(1, 2, 3, 4), keepdim=True)
        zb = zb.mean(dim=(1, 2, 3, 4), keepdim=True)

        # compute scales between 0.5 and 1.5
        alpha = torch.sigmoid(za) + 0.5
        beta = torch.sigmoid(zb) + 0.5

        # compute new window boundaries
        pred_min = alpha * self.min
        pred_max = beta * self.max

        # apply windowing to get intensities between -1 and 1
        x = x.clip(pred_min, pred_max)
        x = (x - pred_min) / (pred_max - pred_min)
        x = (x * 2) - 1

        return x


class GrayscaleAdaptivePerceptionModule(nn.Module):

    def __init__(
        self,
        num_awms: int=3,
        awm_channels: int=16,
        init_level: float=450.0,  # 40.5,
        init_width: float=1100.0,  # 350.0,
    ) -> None:
        super().__init__()

        self.awms = nn.ModuleList()
        for _ in range(num_awms):
            awm = AdaptiveWindowingModule(
                out_channels=awm_channels,
                init_level=init_level,
                init_width=init_width,
            )
            self.awms.append(awm)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'C', 'D', 'H', 'W', torch.float32]:
        xs = []
        for awm in self.awms:
            xs.append(awm(x))

        return torch.cat(xs, dim=1)
