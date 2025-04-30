"""
video_io.py
===========

• preprocess_videos_with_mapping
• extract_videos
• download_files

All functions are **self‑contained** and use only the Python 3 std‑lib
plus numpy / opencv‑python for frame work.
"""

import os
import csv
import glob
import hashlib
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 1.  Frame‑level pre‑processing
# ----------------------------------------------------------------------
def preprocess_videos_with_mapping(
    video_folder: str,
    sequence_length: int,
    height: int,
    width: int,
    channels: int = 3,
    stride: int = 1,
    rgb: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Eagerly loads clips of `sequence_length` consecutive frames (with the
    requested `stride`) from every video in `video_folder`.

    Parameters
    ----------
    rgb : bool
        If True (default) convert BGR→RGB.  Set False to keep OpenCV BGR.
    stride : int
        Sliding‑window step; IV‑VAE uses tc=4 so `stride=4` is handy.

    Returns
    -------
    clips : np.ndarray, shape = (N, L, H, W, C), dtype=float32, range [0,1]
    filenames : list[str]
        Maps each clip to its source video.
    """
    assert channels in (1, 3), "`channels` must be 1 or 3"
    video_sequences, sequence_filenames = [], []

    for video_file in sorted(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            if channels == 1:                       # grayscale if requested
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]
            elif rgb:                               # BGR → RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        # generate sliding‑window clips
        for i in range(0, len(frames) - sequence_length + 1, stride):
            clip = frames[i : i + sequence_length]
            video_sequences.append(clip)
            sequence_filenames.append(video_file)

    clips = np.asarray(video_sequences, dtype=np.float32)  # (N, L, H, W, C)
    return clips, sequence_filenames


# ----------------------------------------------------------------------
# 2.  Cross‑platform selective extractor (pure Python)
# ----------------------------------------------------------------------
def extract_videos(
    csv_path: str,
    zip_base_path: str,
    target_folder: str,
    start_index: int = 0,
    max_videos: int | None = None,
) -> None:
    """
    Extracts specific video files from multi‑gigabyte OpenVid shards
    using the mapping CSV.  Uses `zipfile` so it works on Windows,
    macOS, Linux alike (no external `unzip` dependency).
    """
    rows = list(csv.DictReader(open(csv_path, newline="")))
    rows = rows[start_index : start_index + max_videos if max_videos else None]

    os.makedirs(target_folder, exist_ok=True)

    for row in rows:
        zip_path = Path(zip_base_path) / row["zip_file"]
        member   = row["video_path"]

        if not zip_path.exists():
            print(f"[!] missing shard: {zip_path}")
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                if member not in zf.namelist():
                    print(f"[!] {member} not found inside {zip_path}")
                    continue
                zf.extract(member, path=target_folder)
                print(f"[+] extracted {member}")
        except zipfile.BadZipFile as e:
            print(f"[!] corrupt shard {zip_path}: {e}")


# ----------------------------------------------------------------------
# 3.  Robust shard‑downloader (multi‑part, checksum aware)
# ----------------------------------------------------------------------
def _sha256(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def download_files(output_directory: str, num_parts: int = 1) -> None:
    """
    Downloads OpenVid_part{i}.zip (or its split parts) into
    `<output_directory>/download/` and extracts all videos into
    `<output_directory>/video/`.  CSV mapping files are also fetched.

    Parameters
    ----------
    num_parts : int
        Number of main shards to fetch (OpenVid_part0.zip … part{num_parts-1}.zip)
    """
    root = Path(output_directory)
    zip_dir   = root / "download"
    video_dir = root / "video"
    zip_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"

    for i in range(num_parts):
        full_zip = zip_dir / f"OpenVid_part{i}.zip"
        if full_zip.exists():
            print(f"[✓] {full_zip.name} already present.")
            continue

        # 1️⃣ try to fetch the monolithic zip
        url = f"{base}/OpenVid_part{i}.zip"
        ok  = subprocess.call(["wget", "-q", "-O", full_zip, url]) == 0

        # 2️⃣ fallback to split archives if monolithic fetch failed
        if not ok or full_zip.stat().st_size < 1024:          # very tiny → bad
            print(f"[!] fetching split parts for part{i} ...")
            for part in ("aa", "ab", "ac", "ad"):             # future‑proof
                part_url  = f"{base}/OpenVid_part{i}_part{part}"
                part_file = zip_dir / f"OpenVid_part{i}_part{part}"
                if part_file.exists():
                    continue
                subprocess.run(["wget", "-q", "-O", part_file, part_url])

            # cat them together
            with open(full_zip, "wb") as fout:
                for p in sorted(zip_dir.glob(f"OpenVid_part{i}_part??")):
                    fout.write(open(p, "rb").read())

        # 3️⃣ sha256 check if checksum file is provided (optional)
        sha_url  = f"{base}/OpenVid_part{i}.zip.sha256"
        sha_path = zip_dir / f"OpenVid_part{i}.zip.sha256"
        if subprocess.call(["wget", "-q", "-O", sha_path, sha_url]) == 0:
            want = open(sha_path).read().split()[0]
            have = _sha256(full_zip)
            assert want == have, f"SHA mismatch for part{i}"

        # 4️⃣ finally extract videos (no duplication check here)
        with zipfile.ZipFile(full_zip) as zf:
            zf.extractall(video_dir)
        print(f"[+] extracted part{i} to {video_dir}")

    # 5️⃣ grab mapping CSVs once
    csv_dir = root / "data" / "train"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name in ("OpenVid-1M.csv", "OpenVidHD.csv"):
        dst = csv_dir / name
        if dst.exists():
            continue
        subprocess.run(
            ["wget", "-q", "-O", dst, f"{base}/data/train/{name}"], check=True
        )
        print(f"[+] downloaded {name}")


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Utility helpers for OpenVid pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="download OpenVid shards")
    dl.add_argument("--out", default="./videos", help="output directory")
    dl.add_argument("--parts", type=int, default=1, help="# top‑level shards")

    ex = sub.add_parser("extract", help="selective extractor")
    ex.add_argument("--csv", required=True, help="mapping CSV")
    ex.add_argument("--zip_base", default=".", help="where shards live")
    ex.add_argument("--target", required=True)
    ex.add_argument("--start", type=int, default=0)
    ex.add_argument("--max", type=int)

    args = parser.parse_args()

    if args.cmd == "download":
        download_files(args.out, args.parts)
    elif args.cmd == "extract":
        extract_videos(
            args.csv, args.zip_base, args.target, args.start, args.max
        )
        
        # ----------------------------------------------------------------------
# 4.  Lazy Dataset for PyTorch training (B,T,C,H,W)
# ----------------------------------------------------------------------
import random, torch
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    """
    Lazily decodes videos from `root` and returns a tensor of shape
    (seq_len, 3, H, W) in the 0‑1 range.
    """
    def __init__(self, root: str, seq_len: int = 16, size: tuple[int,int]=(256,256)):
        super().__init__()
        self.paths = [str(Path(root)/f) for f in sorted(os.listdir(root))
                      if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not self.paths:
            raise RuntimeError(f"No video files found in {root}")
        self.seq_len = seq_len
        self.size    = size  # (H, W)

    def __len__(self): return len(self.paths)

    def _decode_resize(self, path: str):
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ok, frame = cap.read()
            if not ok: break
            # ---- resize & colour convert ----
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2,0,1))  # C,H,W
        cap.release()
        return torch.stack(frames).float().div_(255.0)              # T,C,H,W

    def __getitem__(self, idx):
        vid = self._decode_resize(self.paths[idx])  # T,C,H,W
        if vid.size(0) < self.seq_len:
            raise ValueError(f"Video too short: {self.paths[idx]}")
        start = random.randint(0, vid.size(0) - self.seq_len)
        clip  = vid[start : start + self.seq_len]   # L,C,H,W
        return clip

