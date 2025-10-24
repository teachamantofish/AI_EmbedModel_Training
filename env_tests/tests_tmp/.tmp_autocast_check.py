import torch
from torch import amp
m = torch.nn.Linear(1024,1024).cuda()
x = torch.randn(2,1024).cuda()
with amp.autocast(device_type='cuda'):
    y = m(x)
print('forward dtype:', y.dtype)
print('any_half:', y.dtype==torch.float16)
