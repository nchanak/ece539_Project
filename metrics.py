import os
import cv2
import torch
import lpips
import random
import numpy as np
import tensorflow as tf

def compute_video_metrics(original_rgb=None, reconstructed_rgb=None, video_path=None, frame_width=None, max_val=255.0):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    # Set up LPIPS model
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.eval()

    # just fucking look at the original mp4, only looks at one though
    if video_path:
        assert frame_width is not None, "Must provide frame_width when using video_path"
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig = rgb[:, :frame_width]
            recon = rgb[:, frame_width:]

            psnr_scores.append(tf.image.psnr(orig.astype(np.float32), recon.astype(np.float32), max_val=max_val).numpy())
            ssim_scores.append(tf.image.ssim(orig.astype(np.float32), recon.astype(np.float32), max_val=max_val).numpy())

            torch_orig = torch.tensor(orig / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
            torch_recon = torch.tensor(recon / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                lpips_val = loss_fn(torch_orig, torch_recon).item()
            lpips_scores.append(lpips_val)
        cap.release()

    # can also hand it the sequence tensors directly, just make sure right format, ie not normalized
    # Will go through full sequence list
    elif original_rgb is not None and reconstructed_rgb is not None:
        num_sequences, seq_len = original_rgb.shape[:2]
        for i in range(num_sequences):
            for j in range(seq_len):
                orig = original_rgb[i, j].astype(np.float32)
                recon = reconstructed_rgb[i, j].astype(np.float32)

                psnr_scores.append(tf.image.psnr(orig, recon, max_val=max_val).numpy())
                ssim_scores.append(tf.image.ssim(orig, recon, max_val=max_val).numpy())

                torch_orig = torch.tensor(orig / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
                torch_recon = torch.tensor(recon / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float()

                with torch.no_grad():
                    lpips_val = loss_fn(torch_orig, torch_recon).item()
                lpips_scores.append(lpips_val)
    else:
        raise ValueError("Either original/reconstructed arrays or video_path must be provided.")

    return {
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores),
        "lpips": np.mean(lpips_scores),
    }

# Evaluate metrics by video, can specify train/val split and max number of videos to evaluate
# for performance reasons. Randomly samples from split, can specify seed for reproducibility.

def evaluate_metrics_by_video(
    split="train",
    limit=None,
    video_data=None,
    video_filenames=None,
    autoencoder=None,
    sequence_length=None,
    height=None,
    width=None,
    channels=3,
    max_val=255.0,
    seed=42
):
    assert split in ("train", "val")
    assert video_data is not None and video_filenames is not None
    assert autoencoder is not None

    split_index = int(0.8 * len(video_data))
    if split == "train":
        data = video_data[:split_index]
        filenames = video_filenames[:split_index]
    else:
        data = video_data[split_index:]
        filenames = video_filenames[split_index:]

    unique_videos = sorted(set(filenames))

    if limit:
        random.seed(seed)
        unique_videos = random.sample(unique_videos, min(limit, len(unique_videos)))

    results = {}

    for video_name in unique_videos:
        indices = [i for i, name in enumerate(filenames) if name == video_name]
        original = data[indices]

        prediction = autoencoder.predict(original, batch_size=1)
        reconstructed = prediction["reconstruction"] if isinstance(prediction, dict) else prediction

        original_uint8 = (original * 255.0).astype(np.uint8)
        reconstructed_uint8 = (reconstructed * 255.0).astype(np.uint8)

        print(f"📹 Evaluating: {video_name} — {len(indices)} sequences")

        metrics = compute_video_metrics(
            original_rgb=original_uint8,
            reconstructed_rgb=reconstructed_uint8,
            max_val=max_val
        )

        results[video_name] = metrics

    return results

def evaluate_single_video_metrics(
    video_name,
    video_data,
    video_filenames,
    autoencoder,
    sequence_length,
    height,
    width,
    channels=3,
    max_val=255.0
):
    # Locate sequences belonging to this specific video
    indices = [i for i, name in enumerate(video_filenames) if name == video_name]
    if not indices:
        raise ValueError(f"Video name {video_name} not found in dataset.")

    original = video_data[indices]
    prediction = autoencoder.predict(original, batch_size=1)
    reconstructed = prediction["reconstruction"] if isinstance(prediction, dict) else prediction

    original_uint8 = (original * 255.0).astype(np.uint8)
    reconstructed_uint8 = (reconstructed * 255.0).astype(np.uint8)

    print(f"🎞️ Evaluating metrics for: {video_name} — {len(indices)} sequences")

    metrics = compute_video_metrics(
        original_rgb=original_uint8,
        reconstructed_rgb=reconstructed_uint8,
        max_val=max_val
    )

    return metrics