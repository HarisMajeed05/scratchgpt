"""Image tokenizer: a VQ-VAE mapping a 128x128 image to 16x16 = 256 codes from an 8192-entry codebook.

Trained from scratch with L1 + MSE only (no GAN / perceptual loss), so reconstructions are soft; fine detail is lost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths
from ..safeio import load_torch
from ..tokens import IMG_CODES, IMG_DOWN, IMG_SIZE

K, D = IMG_CODES, 64


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1),
                               nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.b(x)


class VQVAE(nn.Module):
    def __init__(self, chs=(128, 128, 256, 256)):
        super().__init__()
        e = [nn.Conv2d(3, chs[0], 3, 1, 1)]
        for i in range(3):                       # 3 downsamples: 128 -> 16
            e += [nn.Conv2d(chs[i], chs[i + 1], 4, 2, 1), Res(chs[i + 1]), Res(chs[i + 1])]
        e += [nn.GroupNorm(32, chs[3]), nn.SiLU(), nn.Conv2d(chs[3], D, 1)]
        self.enc = nn.Sequential(*e)
        d = [nn.Conv2d(D, chs[3], 3, 1, 1), Res(chs[3]), Res(chs[3])]
        for i in (2, 1, 0):
            d += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(chs[i + 1], chs[i], 3, 1, 1),
                  Res(chs[i]), Res(chs[i])]
        d += [nn.GroupNorm(32, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], 3, 3, 1, 1)]
        self.dec = nn.Sequential(*d)
        self.codebook = nn.Parameter(F.normalize(torch.randn(K, D), dim=1))
        self.register_buffer("usage", torch.zeros(K))
        self.h = IMG_SIZE // IMG_DOWN
        self.step = None                         # set by load_vq: which training step this checkpoint is

    def quantize(self, z):
        B = z.shape[0]
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = F.normalize(z.float().permute(0, 2, 3, 1).reshape(-1, D), dim=1)   # cosine-similarity codebook
            cb = F.normalize(self.codebook.float(), dim=1)
            idx = (z @ cb.t()).argmax(1)
            zq = cb[idx]
        return z, zq, idx.view(B, -1)

    def forward(self, x):
        B = x.shape[0]
        z, zq, idx = self.quantize(self.enc(x))
        qloss = 0.25 * F.mse_loss(z, zq.detach()) + F.mse_loss(zq, z.detach())
        zst = z + (zq - z).detach()              # straight-through estimator
        out = self.dec(zst.view(B, self.h, self.h, D).permute(0, 3, 1, 2))
        if self.training:
            self.usage.index_add_(0, idx.flatten(), torch.ones(idx.numel(), device=idx.device))
        return out, qloss, idx, z

    @torch.no_grad()
    def encode(self, x):                         # x: (B,3,H,W) in [-1,1] -> (B,256) code ids in [0,K)
        return self.quantize(self.enc(x))[2]

    @torch.no_grad()
    def decode(self, idx):                       # idx: (B,256) -> (B,3,H,W) in [-1,1]
        B = idx.shape[0]
        zq = F.normalize(self.codebook.float(), dim=1)[idx]
        return self.dec(zq.view(B, self.h, self.h, D).permute(0, 3, 1, 2)).clamp(-1, 1)

    @torch.no_grad()
    def restart_dead(self, z):                   # re-seed unused codes from current encoder outputs
        dead = (self.usage == 0).nonzero().flatten()
        if len(dead):
            self.codebook.data[dead] = z[torch.randint(0, len(z), (len(dead),), device=z.device)]
        self.usage.zero_()
        return len(dead)


def load_vq(path=None, device="cuda"):
    ck = load_torch(path or paths.VQ_CKPT / "vq.pt", device)
    if ck is None:
        raise FileNotFoundError("no trained image tokenizer found. Run: scratchgpt vq-cache, then scratchgpt vq-train")
    m = VQVAE().to(device)
    m.load_state_dict(ck["model"])
    m.step = ck.get("step")                      # lets image data be tied to this exact VQ-VAE version
    return m.eval()
