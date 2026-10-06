import math
import torch
import torch.nn as nn


# ─── Projection Head ────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim=128, num_layers=3):
        super(ProjectionHead, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, mid_dim, bias=True))
                layers.append(nn.BatchNorm1d(mid_dim))
                layers.append(nn.ReLU(inplace=True))
            elif i < num_layers - 1:
                layers.append(nn.Linear(mid_dim, mid_dim, bias=True))
                layers.append(nn.BatchNorm1d(mid_dim))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Linear(mid_dim, out_dim, bias=False))
                layers.append(nn.BatchNorm1d(out_dim, affine=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SupervisedHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(SupervisedHead, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        return self.fc(x)


# ─── LARS Optimizer ─────────────────────────────────────────────────────────

class LARS(torch.optim.Optimizer):
    """Faithful port of google-research/simclr's LARSOptimizer (classic_momentum=True path)."""
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.clone()

                # Weight decay added to grad BEFORE trust ratio (matches official)
                # Exclude 1D params (bias, BN) from weight decay AND layer adaptation,
                # matching official exclude_from_weight_decay / exclude_from_layer_adaptation
                is_excluded = (p.ndim == 1)

                if group['weight_decay'] != 0 and not is_excluded:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Trust ratio: eeta * w_norm / g_norm (g_norm computed on grad AFTER weight decay)
                if is_excluded:
                    trust_ratio = 1.0
                else:
                    w_norm = p.data.norm(2)
                    g_norm = grad.norm(2)
                    if w_norm > 0 and g_norm > 0:
                        trust_ratio = (group['eta'] * w_norm / g_norm).item()
                    else:
                        trust_ratio = 1.0

                scaled_lr = group['lr'] * trust_ratio

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']

                # next_v = momentum * v + scaled_lr * grad  (official classic_momentum, no nesterov)
                buf.mul_(group['momentum']).add_(grad, alpha=scaled_lr)

                p.data.add_(buf, alpha=-1.0)

        return loss


# ─── LR Schedule ────────────────────────────────────────────────────────────

def get_lr_schedule(optimizer, warmup_steps, total_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
