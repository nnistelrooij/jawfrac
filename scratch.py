import torch

ckpt1 = torch.load('checkpoints/old_fractures_linear_displaced_patch_size=64.ckpt')
ckpt2 = torch.load('checkpoints/old_fractures_linear_displaced_patch_size=96.ckpt')

for key in ckpt1['state_dict']:
    if 'cascade' in key:
        continue

    ckpt2['state_dict'][key] = ckpt1['state_dict'][key]

torch.save(ckpt2, 'checkpoints/new_fractures_linear_displaced_patch_size=96.ckpt')
