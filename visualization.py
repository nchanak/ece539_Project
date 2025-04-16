import os
import subprocess
import cv2
import numpy as np
from IPython.display import HTML

# Takes video and generates list of frames in RGB, and another normalized color
def extract_video_frames(video_path, height, width):
    cap = cv2.VideoCapture(video_path)
    raw_rgb_frames = []
    norm_rgb_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        raw_rgb_frames.append(frame_rgb)                     # uint8 RGB
        norm_rgb_frames.append(frame_rgb / 255.0)            # float32 normalized
    cap.release()

    return np.array(raw_rgb_frames), np.array(norm_rgb_frames)

# takes list of frames and generates the sequence list
def create_sequences(frames, sequence_length):
    return np.array([frames[i:i+sequence_length] for i in range(len(frames) - sequence_length + 1)])

def save_comparison_video(original_frames, reconstructed_frames, path="side_by_side.avi", fps=24):
    height, width = original_frames.shape[1], original_frames.shape[2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width * 2, height))

    for orig, rec in zip(original_frames, reconstructed_frames):
        orig_uint8 = orig.astype(np.uint8)
        
        # Flip red/blue in reconstructed output
        rec_uint8 = (np.clip(rec, 0, 1) * 255).astype(np.uint8)
        rec_uint8 = rec_uint8[..., [2, 1, 0]]  # BGR → RGB

        combined = np.hstack((orig_uint8, rec_uint8))  # ✅ uses flipped version now
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


    out.release()
    print(f"[✓] Comparison saved: {path}")


# Specify the video and autoencoder and it'll give you a side-by-side comparison of the downsampled original and the reconstructed video
def generate_video_comparison_by_name(video_name, autoencoder, sequence_length, height, width, video_folder="extracted_videos"):
    video_path = os.path.join(video_folder, video_name)
    if not os.path.exists(video_path):
        print(f"[!] Video file not found: {video_path}")
        return

    # Extract frames (not sequences)
    original_rgb_frames, normalized_rgb_frames = extract_video_frames(
        video_path, height, width
    )

    # Create sequences for prediction
    input_sequences = create_sequences(normalized_rgb_frames, sequence_length)

    # Predict with autoencoder
    reconstructed_sequences = autoencoder.predict(input_sequences)

    # Reassemble reconstructed video: first frame + last from each sequence
    # Handle dict-style output from custom model
    if isinstance(reconstructed_sequences, dict):
        recon_array = reconstructed_sequences["reconstruction"]
    else:
        recon_array = reconstructed_sequences

    reconstructed_frames = [recon_array[0][0]]
    for i in range(len(recon_array)):
        reconstructed_frames.append(recon_array[i][-1])


    reconstructed_frames = np.array(reconstructed_frames)

    # Crop original to match reconstructed length
    original_cropped = original_rgb_frames[:len(reconstructed_frames)]

    # Save the comparison video
    output_path = f"comparison_{os.path.splitext(video_name)[0]}.avi"
    save_comparison_video(original_cropped, reconstructed_frames, path=output_path)

    return output_path

import os

def generate_and_display_comparison(
    video_name,
    autoencoder,
    sequence_length,
    height,
    width,
    video_folder="extracted_videos",
    scale=4,
    width_display=512,
    cleanup=True,
):
    os.makedirs("comparisons", exist_ok=True)

    # make video comparison
    avi_path = generate_video_comparison_by_name(
        video_name=video_name,
        autoencoder=autoencoder,
        sequence_length=sequence_length,
        height=height,
        width=width,
        video_folder=video_folder
    )

    # put in folder
    base_name = os.path.splitext(os.path.basename(avi_path))[0]
    mp4_path = os.path.join("comparisons", f"{base_name}.mp4")

    # avi -> mp4
    convert_avi_to_mp4(avi_path, mp4_path=mp4_path, scale=scale)

    # display html embed
    display = display_video_inline(mp4_path, width=width_display)

    # clean avi
    if cleanup:
        try:
            os.remove(avi_path)
            print(f"Deleted: {avi_path}")
        except Exception as e:
            print(f"[!] Failed to delete AVI: {e}")

    return display


# This conversion is so we can embed the video in a notebook
def convert_avi_to_mp4(avi_path, mp4_path=None, crf=23, scale=4):
    if mp4_path is None:
        mp4_path = os.path.splitext(avi_path)[0] + ".mp4"
    command = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i", avi_path,
        "-vf", f"scale=iw*{scale}:ih*{scale}",
        "-vcodec", "libx264",
        "-crf", str(crf),
        mp4_path
    ]
    print(f"[▶] Converting {avi_path} → {mp4_path}")
    subprocess.run(command, check=True)
    return mp4_path

# Make video larger inline
def display_video_inline(path, width=512):
    height = width // 2
    return HTML(f"""
    <video width="{width}" height="{height}" controls>
      <source src="{path}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """)

