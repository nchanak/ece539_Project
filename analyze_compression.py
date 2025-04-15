import os
import numpy as np
import tensorflow as tf

def get_kb(path):
    return os.path.getsize(path) / 1024

def analyze_compression_from_video_file(
    video_name,
    autoencoder,
    encoder,
    sequence_length,
    height,
    width,
    channels=3,
    video_folder="extracted_videos",
    save_prefix="output"
):
    import cv2

    video_path = os.path.join(video_folder, video_name)
    if not os.path.exists(video_path):
        print(f"[!] Video not found: {video_path}")
        return

    # === Load and preprocess video ===
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

    # === Create sequences ===
    def create_sequences(frames):
        return [frames[i:i+sequence_length] for i in range(len(frames) - sequence_length + 1)]

    raw_seq = np.array(create_sequences(raw_frames))
    norm_seq = np.array(create_sequences(norm_frames))

    print(f"[▶] Processing {len(norm_seq)} sequences from: {video_name}")

    # === Predict ===
    latent = encoder.predict(norm_seq, batch_size=1)
    reconstructed = autoencoder.predict(norm_seq, batch_size=1)

    # === Save results ===
    np.savez_compressed(f"{save_prefix}_latent.npz", data=latent)
    np.savez_compressed(f"{save_prefix}_original_input.npz", data=raw_seq)
    np.savez_compressed(f"{save_prefix}_reconstructed.npz", data=reconstructed)

    # === Compression Stats ===
    latent_kb = get_kb(f"{save_prefix}_latent.npz")
    input_kb = get_kb(f"{save_prefix}_original_input.npz")

    print(f"\n📦 Compression Results for '{video_name}':")
    print(f"Raw 64x64 input size:    {input_kb:.2f} KB")
    print(f"Latent compressed size:  {latent_kb:.2f} KB")
    print(f"Estimated compression:   {input_kb / latent_kb:.2f}x")

    return {
        "video": video_name,
        "num_sequences": len(norm_seq),
        "input_kb": input_kb,
        "latent_kb": latent_kb,
        "compression_ratio": input_kb / latent_kb
    }
