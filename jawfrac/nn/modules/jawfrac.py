import torch
import torch.nn as nn
from torchtyping import TensorType

from jawfrac.nn.modules.unet import UNet


class FracNet(nn.Module):
    
    def __init__(
        self,
        state_dict_path: str,
        **model_cfg,
    ) -> None:
        super().__init__()

        self.unet = UNet(**model_cfg)
        if state_dict_path:
            checkpoint = torch.load(state_dict_path)
            self.unet.load_state_dict(checkpoint)

        self.head = nn.Conv3d(32, 1, 3, padding=1)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'C', 'D', 'H', 'W', torch.float32]:
        _, x = self.unet(x)
        seg = self.head(x)

        return seg.squeeze(dim=1)
