import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from auto_adv.cell import GradCell, RandCell, AllCell

def normalize(X, mu, std):
    if mu is None or std is None:
        return X
    else:
        return (X - mu)/std

upper_limit, lower_limit = 1,0

class AutoXAdv(nn.Module):
    def __init__(
        self,
        grad_dim,
        emb_dim,
        model_loss,
        mu,
        std,
        loss_type='ce',
        n_steps=5,
        step_size=0.007,
        eps=0.2,
        omega=0.001,
        log=None
    ):
        super().__init__()
        self.grad_dim = grad_dim
        self.emb_dim = emb_dim
        self.n_steps = n_steps
        self.eps = eps
        self.omega = omega
        self.model_loss = model_loss
        self.loss_type = loss_type
        self.log = log or print
        self.mu, self.std = mu, std

        self.cells = []
        for _ in range(n_steps):
            self.cells.append(AllCell(grad_dim, emb_dim, step_size, self.log))
        self.cells = nn.ModuleList(self.cells)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        self.log(f"AutoXAdv Args: eps({eps}), step_size({step_size}), n_step({n_steps}), loss_type({loss_type})")


    def forward(self, i, grad, delta, x, eps, last_grad=None, last_adv=None, hard=True):
        if last_grad is None:
            last_grad = torch.randn_like(grad)

        adv = self.cells[i](
            grad, last_grad, last_adv,
            delta=delta, x=x, eps=eps,
            hard=hard,
        )
        return adv


    def optimize_step(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.mul_(-1)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def optimize(self, model, data, target):
        self.optimizer.zero_grad()

        model.eval()
        model_loss = self.model_loss
        x, y = data.cuda().detach(), target.cuda().detach()
        batch_size = y.size(0)

        delta = torch.zeros_like(x, requires_grad=True)
        y_pred = model(normalize(torch.clamp(x + delta, 0.0, 1.0), self.mu, self.std))
        if self.loss_type == 'ce':
            loss = model_loss(y_pred, y)
        else:
            raise ValueError(f'Wrong model loss {self.loss_type}')
        loss.backward(retain_graph=True)
        grad, last_grad = delta.grad.detach().clone(), None
        last_adv = torch.zeros_like(delta)

        for i in range(self.n_steps):
            adv = self(
                i, grad, last_grad=last_grad, last_adv=last_adv, hard=True,
                delta=delta, x=x, eps=self.eps
            )

            delta = delta + adv
            delta.retain_grad()
            y_pred = model(normalize(torch.clamp(x + delta, 0, 1), self.mu, self.std))
            if self.loss_type == 'ce':
                loss = model_loss(y_pred, y)

            if delta.grad is not None:
                delta.grad.zero_()
            self.zero_grad()
            loss.backward(retain_graph=True)
            grad, last_grad = delta.grad.detach().clone(), grad
            last_adv = adv.clone()
        self.optimize_step()

        return delta.detach()

    def on_epoch_start(self):
        for i, cell in enumerate(self.cells):
            cell.mixed_adv.on_epoch_start()

    def on_epoch_end(self):
        for i, cell in enumerate(self.cells):
            cell.mixed_adv.on_epoch_end()
