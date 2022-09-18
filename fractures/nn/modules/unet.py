import torch
import torch.nn as nn
from torchtyping import TensorType

from mandibles.nn import UNet


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
            self.load_state_dict(checkpoint['state_dict'])

        self.head = nn.Conv3d(32, 1, 3, padding=1)

    def forward(
        self,
        x: TensorType['B', 1, 'D', 'H', 'W', torch.float32],
    ) -> TensorType['B', 'C', 'D', 'H', 'W', torch.float32]:
        _, x = self.unet(x)
        seg = self.head(x)

        return seg.squeeze(dim=1)


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
