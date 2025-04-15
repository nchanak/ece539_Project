import os
import numpy as np
import tensorflow as tf
import io

def get_compressed_kb(array):
    """Returns size in KB of a numpy array saved with compression (in-memory)."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return buffer.getbuffer().nbytes / 1024  # Convert bytes to KB

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
    save=True  # optional disk saving
):
    import cv2

    video_path = os.path.join(video_folder, video_name)
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}")
        return

    # load  preprocess
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    norm_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        raw_frames.append(rgb)
        norm_frames.append(rgb / 255.0)
    cap.release()

    if len(raw_frames) < sequence_length:
        print(f"[!] Not enough frames for sequence_length={sequence_length}")
        return

    def create_sequences(frames):
        return [frames[i:i+sequence_length] for i in range(len(frames) - sequence_length + 1)]

    raw_seq = np.array(create_sequences(raw_frames))
    norm_seq = np.array(create_sequences(norm_frames))

    # prediction
    latent = encoder.predict(norm_seq, batch_size=1)
    reconstructed = autoencoder.predict(norm_seq, batch_size=1)

    # compressed size
    input_kb = get_compressed_kb(raw_seq)
    latent_kb = get_compressed_kb(latent)

    if save:
        np.savez_compressed(f"{save_prefix}_latent.npz", data=latent)
        np.savez_compressed(f"{save_prefix}_original_input.npz", data=raw_seq)
        np.savez_compressed(f"{save_prefix}_reconstructed.npz", data=reconstructed)

    return {
        "video": video_name,
        "num_sequences": len(norm_seq),
        "input_kb": input_kb,
        "latent_kb": latent_kb,
        "compression_ratio": input_kb / latent_kb if latent_kb > 0 else float("inf")
    }

import random

def evaluate_compression_by_video(
    split="train",
    limit=None,
    video_data=None,
    video_filenames=None,
    autoencoder=None,
    encoder=None,
    sequence_length=None,
    height=None,
    width=None,
    channels=3,
    video_folder="extracted_videos",
    seed=42
):
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

    from analyze_compression import analyze_compression_from_video_file
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

