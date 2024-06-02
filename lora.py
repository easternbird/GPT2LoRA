import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
    
class Conv1DLoRA(nn.Module):
    def __init__(self, conv1d: Conv1D, rank, alpha):
        super().__init__()
        self.conv1d = conv1d
        in_dim, out_dim = conv1d.weight.shape
        self.lora = LoRALayer(
            in_dim, out_dim, rank, alpha
        )
        
    def forward(self, x):
        conv1d_x = self.conv1d(x)
        size_out = x.size()[:-1] + (self.conv1d.nf,)
        x = self.lora(x.view(-1, x.size(-1)))
        lora_x = x.view(size_out)
        return conv1d_x + lora_x
