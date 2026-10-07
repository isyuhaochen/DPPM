"""Query-independent memory modules. Inputs: [time, ..., latent_width]."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def calibrate(value, target):
    return value * target / value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


class RPMem(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.proj = nn.Linear(2 * d, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)

    def forward(self, qs):
        h = qs[0].float()
        for q in qs[1:]:
            q = q.float()
            gate = self.proj(torch.cat((h, q), dim=-1)).float().sigmoid()
            h = gate * h + (1 - gate) * q
        return h.unsqueeze(0)


class Evidence(nn.Module):
    def __init__(self, d=512, preserve_norm=True):
        super().__init__()
        self.score = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.score.weight)
        self.preserve_norm = preserve_norm

    def forward(self, qs):
        q = qs.float()
        weights = self.score(F.layer_norm(q, (q.shape[-1],))).softmax(dim=0)
        h = (weights * q).sum(dim=0, keepdim=True)
        if self.preserve_norm:
            target = (weights * q.square()).sum(0, keepdim=True).mean(-1, keepdim=True).sqrt()
            h = h * (target / h.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
        return h


def delta_write(matrix, key, value, rate):
    prediction = (matrix * key.unsqueeze(-1)).sum(dim=-2)
    return matrix + key.unsqueeze(-1) * (rate * (value - prediction)).unsqueeze(-2)


class Delta(nn.Module):
    def __init__(self, d=512, key_dim=8, preserve_norm=False):
        super().__init__()
        self.key = nn.Linear(d, key_dim, bias=False)
        self.rate = nn.Linear(d, d)
        self.read_key = nn.Parameter(torch.zeros(key_dim))
        self.preserve_norm = preserve_norm
        nn.init.normal_(self.key.weight, std=0.04)
        nn.init.zeros_(self.rate.weight)
        nn.init.constant_(self.rate.bias, -1.0)

    def forward(self, qs):
        q = qs.float()
        normalized = F.layer_norm(q, (q.shape[-1],))
        keys = F.normalize(F.elu(self.key(normalized)) + 1, dim=-1)
        rates = self.rate(normalized).sigmoid()
        state = keys[0].unsqueeze(-1) * q[0].unsqueeze(-2)
        normalizer = keys[0].unsqueeze(-1) * torch.ones_like(q[0]).unsqueeze(-2)
        for key, value, rate in zip(keys[1:], q[1:], rates[1:]):
            state = delta_write(state, key, value, rate)
            normalizer = delta_write(normalizer, key, torch.ones_like(value), rate)
        read = self.read_key.softmax(-1)
        h = (state * read[:, None]).sum(-2) / (normalizer * read[:, None]).sum(-2).clamp_min(1e-4)
        if self.preserve_norm:
            target = q.square().mean(dim=(0, -1)).sqrt().unsqueeze(-1)
            h = calibrate(h, target)
        return h.unsqueeze(0)


class DPPM(nn.Module):
    def __init__(
        self, d=512, key_dim=4, branch_calibration=True, final_calibration=True, fixed_mixture=False
    ):
        super().__init__()
        self.evidence = Evidence(d, preserve_norm=branch_calibration)
        self.revision = Delta(d, key_dim, preserve_norm=branch_calibration)
        if fixed_mixture:
            self.register_buffer("mix_logit", torch.zeros(d))
        else:
            self.mix_logit = nn.Parameter(torch.zeros(d))
        self.final_calibration = final_calibration

    def forward(self, qs):
        stable, revision = self.evidence(qs), self.revision(qs)
        alpha = self.mix_logit.sigmoid()
        h = alpha * stable + (1 - alpha) * revision
        if self.final_calibration:
            target = alpha * stable.square() + (1 - alpha) * revision.square()
            h = calibrate(h, target.mean(-1, keepdim=True).sqrt())
        return h


class MatchedRPMem(RPMem):
    """Active rank-four gate correction; exactly matches DPPM's parameter count."""

    def __init__(self, d=512):
        super().__init__(d)
        self.correction = nn.Linear(d, 4)
        nn.init.normal_(self.correction.weight, std=0.04)
        nn.init.zeros_(self.correction.bias)
        self.correction_gain = nn.Parameter(torch.zeros(d))
        x = torch.arange(d, dtype=torch.float32) + 0.5
        basis = torch.cos(math.pi * torch.arange(4)[:, None] * x[None] / d)
        basis[0] *= math.sqrt(1 / d)
        basis[1:] *= math.sqrt(2 / d)
        self.register_buffer("correction_basis", basis)

    def forward(self, qs):
        h = qs[0].float()
        for q in qs[1:]:
            q = q.float()
            correction = self.correction(q - h).tanh() @ self.correction_basis
            gate = (self.proj(torch.cat((h, q), -1)) + self.correction_gain * correction).sigmoid()
            h = gate * h + (1 - gate) * q
        return h.unsqueeze(0)


TRAINABLE_METHODS = (
    "dppm",
    "rpmem",
    "evidence",
    "delta",
    "delta_matched",
    "rpmem_matched",
    "dppm_no_final_calibration",
    "dppm_no_calibration",
    "dppm_fixed",
)


def make_memory(method, d=512):
    factories = {
        "dppm": lambda: DPPM(d),
        "rpmem": lambda: RPMem(d),
        "evidence": lambda: Evidence(d),
        "delta": lambda: Delta(d),
        "delta_matched": lambda: Delta(d, 4, True),
        "rpmem_matched": lambda: MatchedRPMem(d),
        "dppm_no_final_calibration": lambda: DPPM(d, final_calibration=False),
        "dppm_no_calibration": lambda: DPPM(d, branch_calibration=False, final_calibration=False),
        "dppm_fixed": lambda: DPPM(d, fixed_mixture=True),
    }
    if method not in factories:
        raise ValueError(f"Unknown memory {method!r}; choose one of {tuple(factories)}")
    return factories[method]()
