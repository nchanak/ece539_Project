import os
import cv2
import numpy as np
import tensorflow as tf
import time
import constriction

# ─────────────────────────────────────────────────────────────────────────────
# Encode a batch of latent indices with adaptive arithmetic coding
# ─────────────────────────────────────────────────────────────────────────────
def encode_latents_aac(prediction, pixelcnn_prior):
    z_idx = tf.reshape(prediction["z_indices"], [-1]).numpy().astype(np.int32)
    batch, T, H, W = prediction["z_indices"].shape
    total_symbols = batch * T * H * W

    enc = constriction.stream.queue.RangeEncoder()
    z_flat = np.zeros(total_symbols, dtype=np.int32)

    for i in range(total_symbols):
        partial = tf.reshape(z_flat, (batch, T, H, W))
        probs = pixelcnn_prior.prob(partial)
        p_i = tf.reshape(probs, [-1, probs.shape[-1]])[i].numpy().astype(np.float32)

        # Smooth and normalize
        min_prob = 1e-6
        p_i = np.clip(p_i, min_prob, None)
        p_i /= np.sum(p_i)

        model = constriction.stream.model.Categorical(p_i, perfect=False)

        symbol = int(z_idx[i])  # ground truth symbol to encode
        enc.encode(symbol, model)

        # Update reconstructed z_flat
        z_flat[i] = symbol

    return enc.get_compressed()


# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Decode a bit-stream back to quantized latents and return z_q
# ─────────────────────────────────────────────────────────────────────────────
def decompress_video(bitstream, vq_layer, pixelcnn_prior, original_shape):
    batch, T, H, W = original_shape
    total_sym = batch * T * H * W

    dec = constriction.stream.queue.RangeDecoder(bitstream)
    z_flat = np.zeros(total_sym, dtype=np.int32)

    for i in range(total_sym):
        partial = tf.reshape(z_flat, (batch, T, H, W))
        probs = pixelcnn_prior.prob(partial)
        p_i = tf.reshape(probs, [-1, probs.shape[-1]])[i].numpy().astype(np.float32)

        # ADD THIS - Smooth and normalize just like encode_latents_aac
        min_prob = 1e-6
        p_i = np.clip(p_i, min_prob, None)
        p_i /= np.sum(p_i)

        model = constriction.stream.model.Categorical(p_i, perfect=False)
        z_flat[i] = dec.decode(model)

    z_tensor = tf.convert_to_tensor(
        z_flat.reshape(batch, T, H, W), dtype=tf.int32)

    one_hot = tf.one_hot(z_tensor, depth=vq_layer.num_embeddings)
    codebook = tf.transpose(vq_layer.embeddings, [1, 0])
    z_q = tf.tensordot(one_hot, codebook, axes=[[4], [0]])

    return z_q

# ─────────────────────────────────────────────────────────────────────────────


# ——— updated analysis function ———
def analyze_with_aac(
    video_name,
    encoder,
    decoder,
    vq_layer,
    pixelcnn_prior,
    sequence_length,
    height,
    width,
    video_folder="extracted_videos",
    save_prefix="aac"
):
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        f = cv2.resize(frame, (width, height))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)/255.0)
    cap.release()

    # pad or trim so we have an integer number of sequences
    N = len(frames) - sequence_length + 1
    seqs = np.array([frames[i:i+sequence_length] for i in range(N)])
    
    # encode → latents
    z_cont = encoder.predict(seqs, batch_size=1)
    z_q, z_idx = vq_layer(z_cont)
    print("z_q size (bytes):", z_q.numpy().nbytes)

    # ——— 1) AAC bitstream & its true size ———
    bitstream = encode_latents_aac(
    {"quantized": z_q, "z_indices": z_idx},
    pixelcnn_prior=pixelcnn_prior
    )
    bytes_used = bitstream.nbytes
    kb_aac     = bytes_used / 1024.0

    # ——— 2) decompress & reconstruct —— (actual VQ-VAE output) ———
    z_q_dec = decompress_video(bitstream, vq_layer, pixelcnn_prior, z_idx.shape)
    recon   = decoder.predict(z_q_dec)

    return {
      "video": video_name,
      "num_sequences": N,
      "bytes_aac": bytes_used,
      "kb_aac": kb_aac,
      "compression_ratio_aac": (np.prod(seqs.shape) * 1 * 8 * height * width / 8) / kb_aac,
      "reconstruction": recon,       # shape: (N, T, H, W, 3)
    }
    

def print_aac_results(results_dict):
    """
    Pretty-print the AAC analysis results.
    
    Args
    ----
    results_dict : dict
        Output of analyze_with_aac (single video results).
    """
    print(f"\n📝 AAC Compression Results for: {results_dict['video']}\n")
    print(f"  • Number of sequences:     {results_dict['num_sequences']}")
    print(f"  • Total bytes (AAC):        {results_dict['bytes_aac']:.2f} bytes")
    print(f"  • Total size (AAC):         {results_dict['kb_aac']:.2f} KB")
    print(f"  • Compression ratio (AAC):  {results_dict['compression_ratio_aac']:.2f}x")
    print(f"  • Reconstruction shape:     {results_dict['reconstruction'].shape}")
    print("\n✅ Done!\n")


import os
import subprocess
import cv2
import numpy as np
from IPython.display import HTML

# ─────────────────────────────────────────────────────────────────────────────
# Save a side-by-side comparison video (original vs AAC reconstructed)
# ─────────────────────────────────────────────────────────────────────────────
def save_aac_comparison_video(original_frames, aac_recon_frames, path="aac_side_by_side.avi", fps=24):
    height, width = original_frames.shape[1], original_frames.shape[2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width * 2, height))

    for orig, recon in zip(original_frames, aac_recon_frames):
        orig_uint8 = orig.astype(np.uint8)
        recon_uint8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        recon_uint8 = recon_uint8[..., [2, 1, 0]]  # RGB flip

        combined = np.hstack((orig_uint8, recon_uint8))
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"[✓] Side-by-side AAC comparison saved to: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Full function to generate and display AAC comparison
# ─────────────────────────────────────────────────────────────────────────────
def generate_and_display_aac_comparison(
    video_name,
    aac_results_dict,
    height,
    width,
    video_folder="extracted_videos",
    scale=4,
    width_display=512,
    cleanup=True
):
    os.makedirs("comparisons", exist_ok=True)

    # === Step 1: Load original video frames ===
    video_path = os.path.join(video_folder, video_name)
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        original_frames.append(rgb)
    cap.release()

    original_frames = np.array(original_frames)

    # === Step 2: Extract reconstructed frames ===
    recon = aac_results_dict["reconstruction"]  # (N, T, H, W, 3)
    recon_frames = [recon[0][0]]
    for i in range(len(recon)):
        recon_frames.append(recon[i][-1])
    recon_frames = np.array(recon_frames)

    # === Step 3: Match frame counts ===
    num_recon_frames = len(recon_frames)
    original_cropped = original_frames[:num_recon_frames]

    # === Step 4: Save side-by-side comparison AVI ===
    avi_path = f"comparison_aac_{os.path.splitext(video_name)[0]}.avi"
    save_aac_comparison_video(original_cropped, recon_frames, path=avi_path)

    # === Step 5: Convert to MP4 for nice playback ===
    base_name = os.path.splitext(os.path.basename(avi_path))[0]
    mp4_path = os.path.join("comparisons", f"{base_name}.mp4")

    convert_avi_to_mp4(avi_path, mp4_path=mp4_path, scale=scale)

    # === Step 6: Display video inline ===
    display = display_video_inline(mp4_path, width=width_display)

    # Cleanup .avi
    if cleanup:
        try:
            os.remove(avi_path)
            print(f"Deleted temporary file: {avi_path}")
        except Exception as e:
            print(f"[!] Failed to delete AVI: {e}")

    return display

# ─────────────────────────────────────────────────────────────────────────────
# Helper: AVI → MP4 Conversion
# ─────────────────────────────────────────────────────────────────────────────
def convert_avi_to_mp4(avi_path, mp4_path=None, crf=23, scale=4):
    if mp4_path is None:
        mp4_path = os.path.splitext(avi_path)[0] + ".mp4"
    command = [
        "ffmpeg",
        "-y",
        "-i", avi_path,
        "-vf", f"scale=iw*{scale}:ih*{scale}",
        "-vcodec", "libx264",
        "-crf", str(crf),
        mp4_path
    ]
    print(f"[▶] Converting {avi_path} → {mp4_path}")
    subprocess.run(command, check=True)
    return mp4_path

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Display MP4 nicely in notebook
# ─────────────────────────────────────────────────────────────────────────────
def display_video_inline(path, width=512):
    height = width // 2
    return HTML(f"""
    <video width="{width}" height="{height}" controls>
      <source src="{path}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """)

def save_aac_reconstruction_only(
    aac_results_dict,
    video_name,
    out_dir="reconstructions",
    fps=24,
    scale=1
):
    """
    Saves only the AAC reconstructed video (RGB) without comparison.

    Args:
        aac_results_dict: Output of `analyze_with_aac`.
        video_name: Original video name (used for filename).
        out_dir: Folder to save the MP4.
        fps: Video frame rate.
        scale: Resize factor (1 = no scaling).
    """
    os.makedirs(out_dir, exist_ok=True)
    recon = aac_results_dict["reconstruction"]  # (N, T, H, W, 3)

    # Flatten sequence to frames
    recon_frames = [recon[0][0]]
    for i in range(len(recon)):
        recon_frames.append(recon[i][-1])
    recon_frames = np.array(recon_frames)

    height, width = recon_frames.shape[1], recon_frames.shape[2]
    output_path = os.path.join(out_dir, f"{os.path.splitext(video_name)[0]}_recon.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * scale, height * scale)
    )

    for frame in recon_frames:
        rgb_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if scale > 1:
            rgb_uint8 = cv2.resize(rgb_uint8, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        writer.write(rgb_uint8)

    writer.release()
    print(f"[✓] Saved reconstructed AAC video to: {output_path}")

import os
import cv2
import numpy as np

def save_aac_reconstruction_only_avi(
    aac_results_dict,
    video_name,
    out_dir="reconstructions",
    fps=24,
    scale=1
):
    """
    Saves only the AAC reconstructed video (RGB) as an AVI using XVID codec.

    Args:
        aac_results_dict: Output of `analyze_with_aac`.
        video_name: Original video name (used for filename).
        out_dir: Folder to save the AVI.
        fps: Video frame rate.
        scale: Resize factor (1 = no scaling).
    """
    os.makedirs(out_dir, exist_ok=True)
    recon = aac_results_dict["reconstruction"]  # (N, T, H, W, 3)

    # Flatten sequence to frames
    recon_frames = [recon[0][0]]
    for i in range(len(recon)):
        recon_frames.append(recon[i][-1])
    recon_frames = np.array(recon_frames)

    height, width = recon_frames.shape[1], recon_frames.shape[2]
    output_path = os.path.join(out_dir, f"{os.path.splitext(video_name)[0]}_recon.avi")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width * scale, height * scale)
    )

    for frame in recon_frames:
        rgb_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if scale > 1:
            rgb_uint8 = cv2.resize(rgb_uint8, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[✓] Saved reconstruction-only AVI to: {output_path}")

def generate_and_display_aac_comparison_avi(
    video_name,
    aac_results_dict,
    height,
    width,
    video_folder="extracted_videos",
    out_dir="comparisons",
    fps=24,
    scale=1
):
    """
    Save side-by-side comparison AVI using XVID codec. No MP4 conversion or inline playback.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load original video frames
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        original_frames.append(rgb)
    cap.release()
    original_frames = np.array(original_frames)

    # Extract reconstructed frames
    recon = aac_results_dict["reconstruction"]  # (N, T, H, W, 3)
    recon_frames = [recon[0][0]]
    for i in range(len(recon)):
        recon_frames.append(recon[i][-1])
    recon_frames = np.array(recon_frames)
    print(f"Recon frames shape: {recon_frames.shape}")

    # Match frame count
    num_recon_frames = len(recon_frames)
    original_cropped = original_frames[:num_recon_frames]

    avi_path = os.path.join(out_dir, f"comparison_aac_{os.path.splitext(video_name)[0]}.avi")
    writer = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width * scale * 2, height * scale)
    )

    for orig, recon in zip(original_cropped, recon_frames):
        orig_uint8 = orig.astype(np.uint8)
        recon_uint8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        if scale > 1:
            orig_uint8 = cv2.resize(orig_uint8, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
            recon_uint8 = cv2.resize(recon_uint8, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
            recon_uint8 = recon_uint8[..., [2, 1, 0]]  # RGB flip
        combined = np.hstack((orig_uint8, recon_uint8))
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[✓] Saved side-by-side comparison AVI to: {avi_path}")
