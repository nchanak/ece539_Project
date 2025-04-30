# cfg.py --------------------------------------------------------
# 1) Model (load your checkpoint)
from dumbassVAE import IVVAE
import torch

def get_model():
    ckpt_path = r"C:\Users\ryan\ece539_Project\checkpoints\ivvae_epoch10.pth"
    model = IVVAE(z=8, width=64)

    blob = torch.load(ckpt_path, map_location="cpu")

    # Lightning-style or custom training loop?
    if "model_state" in blob:
        state_dict = blob["model_state"]
    elif "state_dict" in blob:  # pytorch-lightning default key
        # Lightning prefixes each key with "model." Strip it off.
        state_dict = {k.replace("model.", "", 1): v for k, v in blob["state_dict"].items()}
    else:
        state_dict = blob  # assume it's already a plain dict

    model.load_state_dict(state_dict, strict=True)
    return model

# 2) DataLoader (wrap test videos)
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, ConvertImageDtype
from torch.utils.data import DataLoader, Dataset
import torch, glob

class FolderOfVideos(Dataset):
    def __init__(self, root='./single_video_test', max_frames=16, size=256):
        self.files = sorted(glob.glob(f"{root}/*.mp4"))
        self.maxf = max_frames
        self.tx = Compose([
            Resize(size, antialias=True),
            CenterCrop(size),
            ConvertImageDtype(torch.float32)  # 0-1 range
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # --- read & trim ----------------------------------------------------
        vid, _, _ = read_video(self.files[idx], pts_unit="sec")  # (T,H,W,C)
        vid = vid[:self.maxf]  # ≤ max_frames
        T = vid.size(0)

        # --- to float and (T,C,H,W) ------------------------------------------
        vid = vid.permute(0, 3, 1, 2).float() / 255.0  # (T,C,H,W)

        # --- spatial transforms, frame by frame ------------------------------
        frames = [self.tx(vid[t]) for t in range(T)]  # each (C,H,W)
        vid = torch.stack(frames, dim=0)  # (T,C,H,W)

        # --- pad/loop if clip too short --------------------------------------
        if vid.size(0) < self.maxf:
            repeat = self.maxf // vid.size(0) + 1
            vid = vid.repeat(repeat, 1, 1, 1)[:self.maxf]

        # --- reorder to (C,T,H,W) for IV-VAE ---------------------------------
        vid = vid.permute(1, 0, 2, 3)  # (C,T,H,W)
        return vid

def get_loader():
    return DataLoader(
        FolderOfVideos(),
        batch_size=4,
        shuffle=False,
        num_workers=0,  # <-- CHANGED THIS LINE to avoid multiprocessing pickling errors
        pin_memory=True
    )
