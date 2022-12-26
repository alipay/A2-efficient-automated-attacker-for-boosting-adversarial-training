import torch
import torch.nn as nn
import numpy as np


class FGM(nn.Module):
    def __init__(self) -> None:
        super(FGM, self).__init__()
    
    def forward(self, grad, last_grad, last_adv):
        out = grad.clone()
        b, n, w, h = out.size()
        out_norm = out.view(b, n, w * h).norm(dim=-1).view(b, n , 1, 1) 
        out_norm = torch.where(out_norm != 0, out_norm, torch.ones_like(out_norm))
        out = out * np.sqrt(w * h) / out_norm
        out = out.detach()
        return out


class FGSM(nn.Module):
    def __init__(self) -> None:
        super(FGSM, self).__init__()
    
    def forward(self, grad, last_grad, last_adv):
        out = grad.sign().clone()
        return out


class Gaussian(nn.Module):
    def __init__(self) -> None:
        super(Gaussian, self).__init__()
    
    def forward(self, grad, last_grad, last_adv):
        out = torch.randn_like(grad)

        b, n, w, h = out.size()
        out_norm = out.view(b, n, w * h).norm(dim=-1).view(b, n , 1, 1) 
        out_norm = torch.where(out_norm != 0, out_norm, torch.ones_like(out_norm))
        out = out * np.sqrt(w * h) / out_norm
        out = out.detach()
        return out

class Uniform(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, grad, last_grad, last_adv):
        out = torch.zeros_like(grad)
        out.uniform_(-1, 1)
        
        b, n, w, h = out.size()
        out_norm = out.view(b, n, w * h).norm(dim=-1).view(b, n , 1, 1) 
        out_norm = torch.where(out_norm != 0, out_norm, torch.ones_like(out_norm))
        out = out * np.sqrt(w * h) / out_norm
        out = out.detach()
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, grad, last_grad, last_adv):
        out = torch.zeros_like(grad)
        return out


class FGMM(nn.Module):
    def __init__(
        self,
        beta=0.25
    ) -> None:
        super(FGMM, self).__init__()
        self.beta = beta

    def forward(self, grad, last_grad, last_adv):
        momentum = self.beta * last_grad + grad * (1 - self.beta)
        out = momentum.clone()
        b, n, w, h = out.size()
        out_norm = out.view(b, n, w * h).norm(dim=-1).view(b, n , 1, 1) 
        out_norm = torch.where(out_norm != 0, out_norm, torch.ones_like(out_norm))
        out = out * np.sqrt(w * h) / out_norm
        out = out.detach()
        return out


class FGSMM(nn.Module):
    def __init__(
        self,
        beta=0.25
    ) -> None:
        super(FGSMM, self).__init__()
        self.beta = beta

    def forward(self, grad, last_grad, last_adv):
        momentum = self.beta * last_grad + grad * (1 - self.beta)
        out = momentum.sign().clone()
        return out


class FGMAdvM(nn.Module):
    def __init__(
        self,
        beta=0.75
    ) -> None:
        super(FGMAdvM, self).__init__()
        self.beta = beta

    def forward(self, grad, last_grad, last_adv):
        adv = grad.clone().sign()
        momentum = self.beta * adv + (1 - self.beta) * last_adv
        out = momentum
        return out    
