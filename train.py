# train.py -------------------------------------------------------------
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# your own modules
from dumbassVAE import IVVAE, elbo_loss
from NewVideoIo  import ClipDataset   

# ---- Config ----
train_root = "./extracted_videos/extracted_videos"
batch_size     = 1
seq_len        = 16
frame_size     = (256, 256)             # (H, W)
z_dim          = 8
width          = 64
tc             = 4
lr             = 1e-4
max_epochs     = 10
checkpoint_dir = "./checkpoints2"
log_interval   = 50                     # steps

# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset & Loader
train_set = ClipDataset(root=train_root, seq_len=seq_len, size=frame_size)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# Model, Optimizer, Scaler
model  = IVVAE(z=z_dim, width=width, tc=tc).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95))
scaler = GradScaler()

# ---- Training Loop ----
for epoch in range(1, max_epochs+1):
    model.train()
    running_loss = 0.0

    for step, clip in enumerate(train_loader, start=1):
        # clip: (B, T, C, H, W) → (B, C, T, H, W)
        x = clip.permute(0,2,1,3,4).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with autocast():
            recon, mu, logv = model(x)
            loss, logs      = elbo_loss(x, recon, mu, logv)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item() * x.size(0)

        if step % log_interval == 0:
            avg = running_loss / (step * batch_size)
            print(f"Epoch {epoch:02d} | Step {step:04d} | "
                  f"ELBO {avg:.4f} | L1 {logs['l1'].item():.4f} | "
                  f"LPIPS {logs['lpips'].item():.4f}")

    epoch_loss = running_loss / len(train_set)
    print(f"--- Epoch {epoch:02d} complete | Avg ELBO {epoch_loss:.4f} ---")

    # ---- Checkpoint ----
    ckpt = {
        'epoch'       : epoch,
        'model_state' : model.state_dict(),
        'opt_state'   : opt.state_dict(),
        'scaler_state': scaler.state_dict(),
    }
    torch.save(ckpt, os.path.join(checkpoint_dir, f"ivvae_epoch{epoch:02d}.pth"))
    print(f"Saved checkpoint: {checkpoint_dir}/ivvae_epoch{epoch:02d}.pth\n")

   

print("Training complete.")
