import torch
import torch.nn as nn
import numpy as np
from auto_adv.genotypes import *
from .sample import gumbel_softmax


EPS = 1E-20

class MixedAdv(nn.Module):
    def __init__(
        self,
        primitives,
        primitives_val,
        lrs,
        lrs_val,
        grad_dim,
        emb_dim,
        step_size,
        log,
        temperature=0.1,
    ):
        super(MixedAdv, self).__init__()
        self._ops = []
        for op in primitives:
            self._ops.append(op())
        self._ops = nn.ModuleList(self._ops)
        self._lrs = nn.Parameter(
            torch.tensor(lrs),
            requires_grad=False
        )

        self.alpha_op_emb = nn.Embedding(
            len(primitives), embedding_dim=emb_dim
        )
        self.alpha_op_linear = nn.Linear(grad_dim * 2, emb_dim)

        self.alpha_lr_emb = nn.Embedding(
            len(lrs), embedding_dim=emb_dim
        )
        self.alpha_lr_linear = nn.Linear(grad_dim * 2, emb_dim)

        self.grad_dim = grad_dim
        self.n_ops, self.n_lrs = len(primitives), len(lrs)
        self.primitives_val = primitives_val
        self.lrs_val = lrs_val
        self.step_size = step_size
        self.log = log
        self.temperature = temperature

    
    def forward(self, grad, last_grad, last_adv, delta, x, eps, hard=True):
        grad_w = torch.cat([
            last_grad.view(-1, self.grad_dim), grad.view(-1, self.grad_dim)
            ],
            dim=-1
        )
        alpha_op = (self.alpha_op_emb.weight @ self.alpha_op_linear(grad_w).T).mean(dim=-1)
        alpha_op = gumbel_softmax(alpha_op.unsqueeze(0)).squeeze()

        alpha_lr = (self.alpha_lr_emb.weight @ self.alpha_lr_linear(grad_w).T).mean(dim=-1)
        alpha_lr = (alpha_lr / self.temperature).softmax(dim=-1)

        lr = self._lrs @ alpha_lr
        adv = self.adv_forward(
            grad, last_grad, last_adv, alpha_op,
            hard=hard
        )

        out = lr.item() * adv * eps
        out = torch.clamp(x + torch.clamp(delta + out, -eps, eps), 0., 1.) - delta - x
        out = out.detach()
        
        for w in alpha_op:
            if w.item():
                out = w * out
        out = out * lr / lr.item()
        return out


    def adv_forward(self, grad, last_grad, last_adv, alpha, hard=True):
        if hard:
            out = [op(grad, last_grad, last_adv) for w, op in zip(alpha, self._ops) if w.detach().item()]
        else:
            out = [w * op(grad, last_grad, last_adv) for w, op in zip(alpha, self._ops)]
        return sum(out)

    
    def on_epoch_start(self):
        pass
    
    def on_epoch_end(self):
        pass


class AllCell(nn.Module):
    def __init__(
        self,
        grad_dim,
        emb_dim,
        step_size,
        log
    ):
        super(AllCell, self).__init__()
        self.mixed_adv = MixedAdv(
            PRIMITIVES,
            PRIMITIVES_VAL,
            LRS,
            LRS_VAL,
            grad_dim,
            emb_dim,
            step_size,
            log
        )

    def forward(self, grad, last_grad, last_adv, delta, x, eps, hard=True):
        adv = self.mixed_adv(grad, last_grad, last_adv, delta, x, eps, hard=hard)
        return adv


class GradCell(nn.Module):
    def __init__(
        self,
        grad_dim,
        emb_dim,
        log
    ):
        super(GradCell, self).__init__()
        self.mixed_adv = MixedAdv(
            GRAD_PRIMITIVES,
            GRAD_PRIMITIVES_VAL,
            LRS,
            LRS_VAL,
            grad_dim,
            emb_dim,
            log
        )

    def forward(self, grad, last_grad, last_adv, hard=True):
        adv = self.mixed_adv(grad, last_grad, last_adv, hard=hard)
        return adv
    
    
class RandCell(nn.Module):
    def __init__(
        self,
        grad_dim,
        emb_dim,
        log
    ):
        super(RandCell, self).__init__()
        self.mixed_adv = MixedAdv(
            RAND_PRIMITIVES,
            RAND_PRIMITIVES_VAL,
            LRS,
            LRS_VAL,
            grad_dim,
            emb_dim,
            log
        )

    def forward(self, grad, last_grad, last_adv, hard=True):
        adv = self.mixed_adv(grad, last_grad, last_adv, hard=hard)
        return adv


class AdvCell(nn.Module):
    def __init__(
        self,
    ):
        global LRS
        super(AdvCell, self).__init__()
        self.mixed_adv = MixedAdv()
        self.lrs = nn.Parameter(
            torch.tensor(LRS),
            requires_grad=False
        )
        

    def forward(self, grad, last_grad, alpha, alpha_lr, epsilon=None, hard=True):
        lr = self.lrs @ alpha_lr
        adv = self.mixed_adv(
            grad, last_grad, alpha, hard=hard
        )
        
        b, n, w, h = adv.size()
        out = adv * np.sqrt(w * h) / adv.view(b, n, w * h).norm(dim=-1).view(b, n , 1, 1)
        out = lr * adv
        return out
