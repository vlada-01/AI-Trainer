from torch import nn
import torch
import torch.nn.functional as F

# TODO: add BatchNorm2d

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=0, stride=1):
        super().__init__()

        self.h, self.w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.kernels = nn.Linear(in_channels * self.h * self.w, out_channels)
        
    def forward(self, x):
        if self.padding > 0:
            p = self.padding
            x = F.pad(x, [p, p, p, p])
        B, C, _, _ = x.size()

        x = x.unfold(2, self.h, self.stride)
        x = x.unfold(3, self.w, self.stride)
        _, _, H_out, W_out, _, _ = x.size()
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(B, H_out, W_out, C * self.h * self.w)

        out = self.kernels(x)
        return out.permute(0, 3, 1, 2)
    
class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, training=self.training)
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=1, end_dim=-1)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        if self.bias is not None:
            bound = 1 / self.weight.size(1) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=(2,2), stride=2):
        super().__init__()
        
        h, w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.h = h
        self.w = w
        self.stride = stride

    def forward(self, x):
        B, C, _, _ = x.size()

        x = x.unfold(2, self.h, self.stride).unfold(3, self.w, self.stride)
        _, _, H_out, W_out, _, _ = x.size()
        x = x.reshape(B, C, H_out, W_out, self.h * self.w)
        out, _ = torch.max(x, dim=4)
        return out
    
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(x, min=0)
    

        

    
