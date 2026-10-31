"""Optional reviewed MLP24 / Patch3 recipes. Import requires the neural extra."""
import numpy as np
import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.n1 = nn.LayerNorm(64); self.attn = nn.MultiheadAttention(64, 4, dropout=.1, batch_first=True)
        self.n2 = nn.LayerNorm(64); self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64), nn.Dropout(.1))

    def forward(self, x):
        z = self.n1(x); x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class MLP24(nn.Module):
    def __init__(self, width):
        super().__init__(); self.net = nn.Sequential(nn.Flatten(1), nn.Linear(24 * 2 * width, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 24))

    def forward(self, x): return self.net(x)


class Patch3(nn.Module):
    def __init__(self, width):
        super().__init__(); self.width = width; self.project = nn.Linear(3 * 2 * width, 64)
        self.position = nn.Parameter(torch.empty(1, 8, 64)); nn.init.normal_(self.position, std=.02)
        self.blocks = nn.Sequential(AttentionBlock(), AttentionBlock()); self.norm = nn.LayerNorm(64); self.head = nn.Linear(64, 3)

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.project(x.reshape(-1, 8, 3 * 2 * self.width)) + self.position))).reshape(-1, 24)


def factory(name, width): return MLP24(width) if name == "mlp24" else Patch3(width)


def scaler(x):
    width = x.shape[-1]; count = np.zeros(width); total = count.copy(); square = count.copy()
    for i in range(0, len(x), 512):
        a = np.asarray(x[i:i+512], dtype="float64").reshape(-1, width); good = np.isfinite(a); a = np.where(good, a, 0)
        count += good.sum(0); total += a.sum(0); square += (a*a).sum(0)
    mean = total / np.maximum(count, 1); std = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean**2, 0)); std[std < 1e-6] = 1
    return {"mean": mean.astype("float32"), "std": std.astype("float32")}


def transform(x, state):
    missing = ~np.isfinite(x); values = np.clip((np.where(missing, state["mean"], x) - state["mean"]) / state["std"], -10, 10)
    return np.concatenate([values, missing.astype("float32")], axis=-1).astype("float32")


def predict(net, x, state, device):
    net.eval(); result = np.empty((len(x), 24), dtype="float64")
    with torch.no_grad():
        for i in range(0, len(x), 256): result[i:i+256] = net(torch.from_numpy(transform(x[i:i+256], state)).to(device)).cpu().numpy()
    return np.maximum(result, 0)
