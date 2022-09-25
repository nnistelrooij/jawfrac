import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.unet import UNet


class FracNet(nn.Module):
    
    def __init__(
        self,
        **model_cfg,
    ) -> None:
        super().__init__()

        self.unet = UNet(**model_cfg)
        self.head = nn.Conv3d(32, 1, 3, padding=1)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'D', 'H', 'W', torch.float32]:
        _, x = self.unet(x)
        seg = self.head(x)

        return seg.squeeze(dim=1)
