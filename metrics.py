import tensorflow as tf
import torch
import lpips
import numpy as np

def compute_video_metrics(original_rgb, reconstructed_rgb, max_val=255.0):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    loss_fn = lpips.LPIPS(net='alex')  # You can also use 'vgg' or 'squeeze'
    loss_fn.eval()
    num_sequences, seq_len = original_rgb.shape[:2]

    for i in range(num_sequences):
        for j in range(seq_len):
            orig = original_rgb[i, j].astype(np.float32)
            recon = reconstructed_rgb[i, j].astype(np.float32)

            tf_orig = tf.convert_to_tensor(orig, dtype=tf.float32)
            tf_recon = tf.convert_to_tensor(recon, dtype=tf.float32)

            psnr = tf.image.psnr(tf_orig, tf_recon, max_val=max_val).numpy()
            ssim = tf.image.ssim(tf_orig, tf_recon, max_val=max_val).numpy()

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            torch_orig = torch.tensor(orig / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            torch_recon = torch.tensor(recon / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                lpips_val = loss_fn(torch_orig, torch_recon).item()
            lpips_scores.append(lpips_val)

    return {
        "psnr": np.mean(psnr_scores),
        "ssim": np.mean(ssim_scores),
        "lpips": np.mean(lpips_scores),
    }
