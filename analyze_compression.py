import os
import cv2
import numpy as np
import tensorflow as tf
import io
import time

def get_compressed_kb(array):
    """Returns size in KB of a numpy array saved with compression (in-memory)."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return buffer.getbuffer().nbytes / 1024  # Convert bytes to KB

def estimate_compression_bits(z_indices, probs):
    """Estimate number of bits using Shannon entropy of the prior distribution."""
    flattened = z_indices.numpy().flatten()
    prob_array = probs.numpy()
    bits = np.sum(-np.log2(prob_array[flattened] + 1e-9))
    return bits / 8 / 1024  # Convert bits to KB

def estimate_input_kb(norm_seq):
    """Estimate compressed size of original input using entropy of pixel distribution."""
    pixels = norm_seq.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0.0, 1.0))
    prob = hist / np.sum(hist) + 1e-9  # avoid log(0)
    entropy_bits = -np.sum(prob * np.log2(prob))
    total_pixels = pixels.size
    estimated_bits = total_pixels * entropy_bits
    return estimated_bits / 8 / 1024  # convert bits to KB

def compress_64x64_with_codec(frames_64x64, output_path, fps=30, fourcc="mp4v"):

    if os.path.exists(output_path):
        os.remove(output_path)

    # frames_64x64 are assumed to be in shape (N, 64, 64, 3) in either RGB or BGR
    # OpenCV expects BGR, so if frames are in RGB, convert below.
    
    # FourCC code, e.g. 'mp4v' or 'avc1' or 'DIVX' etc. 
    # For .mp4 container, 'mp4v' is common. 
    # For .avi container, 'DIVX' or 'XVID' might be used. 
    fourcc_enc = cv2.VideoWriter_fourcc(*fourcc)
    
    # Initialize video writer
    writer = cv2.VideoWriter(output_path, fourcc_enc, fps, (64, 64))

    for frame in frames_64x64:
        # If frames are in RGB, convert to BGR for OpenCV:
        bgr = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2BGR)
        # Or if you have frames as uint8, ensure they are in [0,255]
        # bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Possibly re-scale if your frames are float32 in [0,1]
        # writer expects 8-bit. So:
        bgr = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
        writer.write(bgr)
    writer.release()

    # Measure file size in KB
    file_size_kb = os.path.getsize(output_path) / 1024.0
    return file_size_kb, len(frames_64x64)

def analyze_compression_from_video_file(
    video_name,
    autoencoder,
    encoder,
    sequence_length,
    height,
    width,
    channels=3,
    video_folder="extracted_videos",
    save_prefix="output",
    fps=30,
    codec_fourcc="mp4v",
    save=False  # optional disk saving
):
    import cv2

    video_path = os.path.join(video_folder, video_name)
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    norm_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))  # e.g., 64×64
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        raw_frames.append(rgb)
        norm_frames.append(rgb / 255.0)
    cap.release()

    if len(raw_frames) < sequence_length:
        print(f"[!] Not enough frames for sequence_length={sequence_length}")
        return

    # Build sequences for any sequential model usage
    def create_sequences(frames):
        return [frames[i : i + sequence_length] for i in range(len(frames) - sequence_length + 1)]

    raw_seq = np.array(create_sequences(raw_frames))     # (num_sequences, sequence_length, H, W, 3)
    norm_seq = np.array(create_sequences(norm_frames))   # same shape but normalized

    # Estimate raw input size in KB without actual compression artifact (entropy-based)
    input_kb = estimate_input_kb(norm_seq)

    # Flatten out all frames from all sequences. If you only want the first sequence,
    # you can do something like norm_seq[0], but typically you'd handle all frames:
    all_64x64_frames = np.concatenate(norm_seq, axis=0)  # shape: (N * sequence_length, 64, 64, 3)

    # Where to store the compressed file
    codec_output_path = os.path.join("comparisons", f"{save_prefix}_codec_{video_name}.mp4")

    codec_kb, num_frames_written = compress_64x64_with_codec(
        frames_64x64=all_64x64_frames,
        output_path=codec_output_path,
        fps=fps,
        fourcc=codec_fourcc
    )

    start = time.time()

    # If your encoder can handle (batch_size, sequence_length, H, W, C), do so.
    # Otherwise, flatten or adapt accordingly. For example:
    z_cont = encoder.predict(norm_seq, batch_size=1)  # shape: (num_sequences, sequence_length, latent_H, latent_W, channels)
    
    # Vector-quantize to get discrete indices
    z_quant, z_indices = autoencoder.vq_layer(z_cont)  # e.g. shape of z_indices: (num_sequences, sequence_length, latent_H, latent_W)
    
    # --- Actual latent storage size (raw index size before entropy coding) ---
    # Assume each z_index takes log2(num_embeddings) bits. Or use 8, 16, or 32 bits based on your system/storage method.

    num_latents = np.prod(z_indices.shape)  # total number of indices
    bits_per_index = np.ceil(np.log2(autoencoder.vq_layer.num_embeddings))  # bits per index (e.g., log2(512) = 9)
    latent_kb_raw = (num_latents * bits_per_index) / 8 / 1024  # convert bits to KB


    # If you are using a PixelCNN prior, get the probability distribution
    logits = autoencoder.pixelcnn_prior(z_indices)  # shape: (num_sequences, sequence_length, latent_H, latent_W, 256)
    probs = tf.nn.softmax(logits, axis=-1)
    # Average across all positions if you're estimating a single distribution:
    probs = tf.reduce_mean(probs, axis=(0, 1, 2, 3))  # shape: (256,)

    # Estimate compressed size using Shannon entropy
    latent_kb = estimate_compression_bits(z_indices, probs)
    vq_time = time.time() - start

    return {
        "video": video_name,
        "num_sequences": len(norm_seq),
        "num_frames": len(norm_seq) * sequence_length,
        "input_kb_estimated": input_kb,        # Entropy-based estimate of uncompressed data
        "codec_kb": codec_kb,                  # Actual on-disk size from the traditional codec
        "latent_kb_raw": latent_kb_raw,  # <--- Added
        "latent_kb_estimated": latent_kb,      # Entropy-based estimate from PixelCNN
        "compression_ratio_codec": input_kb / codec_kb if codec_kb > 0 else float("inf"),
        "compression_ratio_latent": input_kb / latent_kb if latent_kb > 0 else float("inf"),
        "vq_time_secs": vq_time
    }

def evaluate_compression_by_video(
    split="train",
    limit=None,
    video_data=None,
    video_filenames=None,
    autoencoder=None,
    encoder=None,
    sequence_length=8,
    height=64,
    width=64,
    channels=3,
    video_folder="extracted_videos",
    seed=42
):
    import random
    assert split in ("train", "val")
    assert all(x is not None for x in [video_data, video_filenames, autoencoder, encoder])

    # Split selection
    split_index = int(0.8 * len(video_data))
    filenames = video_filenames[:split_index] if split == "train" else video_filenames[split_index:]

    # Unique video names in split
    unique_videos = sorted(set(filenames))
    if limit:
        random.seed(seed)
        unique_videos = random.sample(unique_videos, min(limit, len(unique_videos)))

    print(f"Evaluating compression on {len(unique_videos)} {split} videos...\n")

    results = {}

    for video_name in unique_videos:
        try:
            result = analyze_compression_from_video_file(
                video_name=video_name,
                autoencoder=autoencoder,
                encoder=encoder,
                sequence_length=sequence_length,
                height=height,
                width=width,
                channels=channels,
                video_folder=video_folder,
                save_prefix="temp_compression",
                save=False
            )
            if result:
                results[video_name] = result
        except Exception as e:
            print(f"[!] Failed to evaluate '{video_name}': {e}")

    return results