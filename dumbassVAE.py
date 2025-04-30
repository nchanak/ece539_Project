# =============================================================================
# models.py   – IV‑VAE skeleton (PyTorch 2.x, CUDA‑12)
# =============================================================================
# Usage:
#   from models import IVVAE, elbo_loss
#   model = IVVAE(z=8, width=64).cuda()
#   ...
# =============================================================================

# %% ---------------------------------------------------------------- imports
import math, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy

# %% ---------------------------------------------------------------- helpers: normalisation
class RMSNorm(nn.Module):
    """Root‑mean‑square normalisation (causal‑safe)."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps   = eps
    def forward(self, x):
        # works for [B,C,H,W] or [B,C,T,H,W]
        rms = x.pow(2).mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.scale[(None, slice(None)) + (None,)*(x.ndim-2)]

# %% ---------------------------------------------------------------- 2‑D building blocks
def conv3x3(c_in, c_out, stride=1):   # small helper
    return nn.Conv2d(c_in, c_out, 3, stride, 1)

class ResBlock2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv3x3(c, c)
        self.conv2 = conv3x3(c, c)
        self.norm1 = RMSNorm(c); self.norm2 = RMSNorm(c)
        self.act   = nn.GELU()
    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)

class Down2D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.Conv2d(c_in, c_out, 4, 2, 1)   # strided conv
    def forward(self, x):  return self.pool(x)

class Up2D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 4, 2, 1)
    def forward(self, x):  return self.up(x)

# %% ---------------------------------------------------------------- image VAE (2‑D)
class ImageEncoder2D(nn.Module):
    def __init__(self, z_dim: int = 4, width: int = 64):
        super().__init__()
        self.stem  = conv3x3(3, width)
        self.down1 = nn.Sequential(ResBlock2D(width),
                                   Down2D(width, width*2))
        self.down2 = nn.Sequential(ResBlock2D(width*2),
                                   Down2D(width*2, width*4))
        self.mid   = ResBlock2D(width*4)
        self.mu     = nn.Conv2d(width*4, z_dim, 1)
        self.logvar = nn.Conv2d(width*4, z_dim, 1)
    def forward(self, x):
        h = self.mid(self.down2(self.down1(self.stem(x))))
        return self.mu(h), self.logvar(h)

class ImageDecoder2D(nn.Module):
    def __init__(self, z_dim=4, width=64):
        super().__init__()
        self.up1  = nn.Sequential(Up2D(z_dim, width*4),
                                  ResBlock2D(width*4))
        self.up2  = nn.Sequential(Up2D(width*4, width*2),
                                  ResBlock2D(width*2))
        # no further upsample—just map width*2→3 channels
        self.head = nn.Sequential(
            nn.Conv2d(width*2, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.head(self.up2(self.up1(z)))

# %% ---------------------------------------------------------------- 3‑D causal blocks
class GCConv3d(nn.Module):
    """
    Group‑Causal Conv: intra‑group (tc) frames can interact; groups stay causal.
    """
    def __init__(self, c_in, c_out, tc: int = 4):
        super().__init__()
        self.tc   = tc
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)

    def forward(self, x):                       # x: B,C,T,H,W
        B, C, T, H, W = x.shape
        # causal pad
        prev = x[:, :, :1]
        pad  = torch.cat([prev, x], dim=2)[:, :, :-1]  # (B,C,T)

        # group frames into chunks of size tc
        g   = math.ceil(T / self.tc)                 # number of groups
        pad = pad.reshape(B, C, g, self.tc, H, W)    # (B,C,g,tc,H,W)
        pad = pad.permute(0, 2, 1, 3, 4, 5)          # (B,g,C,tc,H,W)
        pad = pad.reshape(B * g, C, self.tc, H, W)   # (B·g,C,tc,H,W)

        # apply intra‑group conv
        y = self.conv(pad)                           # (B·g,C_out,tc,H,W)

        # un‑merge batch & group dims
        y = y.reshape(B, g, -1, self.tc, H, W)       # (B,g,C_out,tc,H,W)
        y = y.permute(0, 2, 1, 3, 4, 5).reshape(B, -1, T, H, W)
        return y                                    # (B,C_out,T,H,W)



# --------------------------------------------------------------------- KTC unit
class KTCBlock(nn.Module):
    """Dual branch: 2‑D (key‑frame) + 3‑D (motion)."""
    def __init__(self, c_in, c_out, tc=4):
        super().__init__()
        half = c_out // 2
        self.conv2 = nn.Conv2d(c_in, half, 3, 1, 1)
        self.conv3 = GCConv3d(c_in, half, tc=tc)
        self.n2 = RMSNorm(half);  self.n3 = RMSNorm(half)
        self.act = nn.GELU()

    def forward(self, x):                      # x [B,C,T,H,W]
        key = self.n2(self.conv2(x[:,:,0]))    # (B,half,H,W)
        key = key.unsqueeze(2).repeat(1,1,x.size(2),1,1)
        mot = self.n3(self.conv3(x))           # (B,half,T,H,W)
        return self.act(torch.cat([key, mot], dim=1))

# %% ---------------------------------------------------------------- helper: inflate 2‑D -> 3‑D
# ----------------------------------------------------------------  inflate helpers

from copy import deepcopy
import torch.nn as nn
import torch

def inflate_conv2d(conv2d: nn.Conv2d) -> nn.Conv3d:
    # central‐slice inflate: W3D = [0,0,W2D]
    w2d = conv2d.weight.data
    w3d = torch.zeros(w2d.size(0), w2d.size(1), 3, *w2d.shape[2:], device=w2d.device)
    w3d[:,:,2].copy_(w2d)
    conv3 = nn.Conv3d(conv2d.in_channels, conv2d.out_channels,
                      (3,*conv2d.kernel_size),
                      (1,*conv2d.stride),
                      (1,*conv2d.padding),
                      bias=(conv2d.bias is not None))
    conv3.weight.data.copy_(w3d)
    if conv2d.bias is not None:
        conv3.bias.data.copy_(conv2d.bias.data)
    return conv3

def inflate_convtranspose2d(conv: nn.ConvTranspose2d) -> nn.ConvTranspose3d:
    # inflate only spatial dims; keep time dim =1
    kx, ky = conv.kernel_size
    sx, sy = conv.stride
    px, py = conv.padding

    conv3 = nn.ConvTranspose3d(conv.in_channels, conv.out_channels,
                               kernel_size=(1, kx, ky),
                               stride=(1, sx, sy),
                               padding=(0, px, py),
                               bias=(conv.bias is not None))
    # central‐slice init: put 2D weights into temporal slice 0
    w2 = conv.weight.data  # shape (in, out, kx, ky)
    w3 = torch.zeros_like(conv3.weight.data)
    w3[:,:,0] = w2
    conv3.weight.data.copy_(w3)
    if conv.bias is not None:
        conv3.bias.data.copy_(conv.bias.data)
    return conv3

def _make_3d(module: nn.Module, tc: int):
    """
    Recursively replace Conv2d→Conv3d and ConvTranspose2d→ConvTranspose3d
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, inflate_conv2d(child))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, inflate_convtranspose2d(child))
        else:
            _make_3d(child, tc)

def inflate_encoder(enc2d: nn.Module, tc: int):
    enc3d = deepcopy(enc2d)
    _make_3d(enc3d, tc)
    return enc3d

def inflate_decoder(dec2d: nn.Module, tc: int):
    dec3d = deepcopy(dec2d)
    _make_3d(dec3d, tc)
    return dec3d



# %% ---------------------------------------------------------------- full IV‑VAE
class IVVAE(nn.Module):
    def __init__(self, z: int = 8, width: int = 64, tc: int = 4):
        super().__init__()
        # 2D key‑frame VAE
        self.enc2d = ImageEncoder2D(z//2, width)
        self.dec2d = ImageDecoder2D(z//2, width)

        # 3D motion VAE: full‑z decoder inflated to 3D
        self.enc3d = inflate_encoder(self.enc2d, tc)
        self.dec3d = inflate_decoder(ImageDecoder2D(z, width), tc)

        # swap ResBlock2D → KTCBlock
        self._ktcify(self.enc3d, tc)
        self._ktcify(self.dec3d, tc)

        self.tc = tc


    # -- utility to swap ResBlock2D -> KTCBlock -----------------------
    @staticmethod
    def _ktcify(module, tc):
        for name, child in module.named_children():
            if isinstance(child, ResBlock2D):
                in_c = out_c = child.conv1.in_channels
                setattr(module, name, KTCBlock(in_c, out_c, tc))
            else:
                IVVAE._ktcify(child, tc)

    # ------------------- encode / reparameterise / decode ------------
    def encode(self, x):                        # x: (B,3,T,H,W)
        # 2D key‑frame path
        mu2, log2 = self.enc2d(x[:, :, 0])      # → (B, Z/2, H, W)
        # 3D motion path
        mu3, log3 = self.enc3d(x)               # → (B, Z/2, T, H, W)

        # Expand the 2D outputs across time
        B, C2, H, W = mu2.shape
        T = x.size(2)
        mu2  = mu2.unsqueeze(2).expand(B, C2, T, H, W)
        log2 = log2.unsqueeze(2).expand(B, C2, T, H, W)

        # Now both are 5D: concatenate on channel axis
        mu     = torch.cat([mu2,  mu3],  dim=1)  # (B, Z, T, H, W)
        logvar = torch.cat([log2, log3], dim=1)
        return mu, logvar

    def decode(self, z):                         # z: (B, Z, T, H, W)
        # split into key‑frame & motion latents
        z2, z3 = torch.chunk(z, 2, dim=1)        # each (B, Z/2, T, H, W)

        # 1) Decode only the 0th time slice in 2D
        key0 = self.dec2d(z2[:, :, 0])           # → (B, 3, H, W)

        # 2) Decode the full video in 3D
        vid = self.dec3d(torch.cat([z2, z3], dim=1))  # → (B, 3, T, H, W)

        # 3) Clone then overwrite the first frame
        vid = vid.clone()
        vid[:, :, 0] = key0                          # both are (B,3,H,W)

        return vid




    def forward(self, x):
        mu, logv = self.encode(x)
        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std)
        z   = mu + eps * std
        xh  = self.decode(z)
        return xh, mu, logv

# %% ---------------------------------------------------------------- loss helper
try:
    import lpips
    _lpips_alex = lpips.LPIPS(net='alex').eval().cuda()
except ModuleNotFoundError:
    _lpips_alex = None

def elbo_loss(x, recon, mu, logv, beta=1e-4):
    l1  = torch.mean(torch.abs(recon - x))
    lp  = torch.zeros_like(l1)
    if _lpips_alex is not None:
        # LPIPS expects BCHW images; flatten time
        B, C, T, H, W = x.shape
        lp = _lpips_alex(recon.permute(0,2,1,3,4).reshape(-1,C,H,W),
                         x.permute(0,2,1,3,4).reshape(-1,C,H,W)).mean()
    kl  = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return l1 + 0.2 * lp + beta * kl, {'l1': l1, 'lpips': lp, 'kl': kl}

# %% ---------------------------------------------------------------- quick sanity test
if __name__ == "__main__":
    B,T,H,W = 2, 16, 256, 256
    x = torch.randn(B, 3, T, H, W).cuda()
    model = IVVAE(z=8, width=64).cuda()
    with torch.cuda.amp.autocast():
        y, mu, logv = model(x)
    print("output", y.shape)                     # should be (B,3,T,H,W)
    loss, log = elbo_loss(x, y, mu, logv)
    print("loss", loss.item(), log)
