# quick_test.py  -------------------------------------------------------
import os
import torch, cv2, numpy as np
from pathlib import Path
from NewVideoIo import ClipDataset          # your video loader
from dumbassVAE import IVVAE, elbo_loss      # your model file


# ---------------------------------------------------------------------
root = './single_video_test'                       # folder with test.mp4
ds   = ClipDataset(root, seq_len=16, size=(256,256))
clip = ds[0].unsqueeze(0)                      # (1, T, C, H, W)
print("loaded clip", clip.shape)

# ---------------------------------------------------------------------
# 2) load model (epoch 10)
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = IVVAE(z=8, width=64).to(device)
ckpt = torch.load("./checkpoints/ivvae_epoch10.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

# ---------------------------------------------------------------------
# 3) run the model
with torch.no_grad():
    x      = clip.permute(0,2,1,3,4).to(device)   # (1, C, T, H, W)
    recon, mu, logv = model(x)
    loss, logs      = elbo_loss(x, recon, mu, logv)
    print(f"ELBO (epoch {ckpt['epoch']}): {loss.item():.4f}")

# ---------------------------------------------------------------------
# 4) save original & reconstructed as .avi
def save_video(tensor, path):
    # tensor (T, C, H, W), RGB float32 [0,1]
    T, C, H, W = tensor.shape
    vw = cv2.VideoWriter(str(path),
                         cv2.VideoWriter_fourcc(*"XVID"),
                         8, (W, H))
    for f in tensor:
        frame = (f.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()

out_dir = Path("./debug_videos"); out_dir.mkdir(exist_ok=True)
# permute back to (T,C,H,W)
save_video(x[0].permute(1,0,2,3).cpu(),    out_dir/"original.avi")
save_video(recon[0].permute(1,0,2,3).cpu(), out_dir/"recon_epoch10trasin1.avi")
print("videos saved to", out_dir.resolve())
