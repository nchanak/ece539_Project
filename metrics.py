import os
import cv2
import torch
import lpips
import numpy as np
import tensorflow as tf

def compute_video_metrics(original_rgb=None, reconstructed_rgb=None, video_path=None, frame_width=None, max_val=255.0):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    # Set up LPIPS model
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.eval()

    # Case 1: use side-by-side video file
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

    # Case 2: use input sequences
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
