from typing import Tuple

from monai.networks.nets import SwinUNETR
import torch
from torchtyping import TensorType


class SwinUNETRBackbone(SwinUNETR):

    def forward(
        self,
        x: TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
    ) -> Tuple[
        TensorType['B', 'h', 'd', 'h', 'w'],
        TensorType['B', 'C^', 'D', 'H', 'W', torch.float32],
    ]:     
        x = x.permute(0, 2, 3, 4, 1)
        hidden_states_out = self.swinViT(x, self.normalize)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        return dec4, out
