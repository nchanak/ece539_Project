import os
import cv2
import numpy as np

def preprocess_videos(video_folder, sequence_length, height, width, channels):
    video_sequences = []
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame to the correct dimensions
            frame = cv2.resize(frame, (width, height))
            # Normalize pixel values to [0, 1]
            frame = frame / 255.0
            frames.append(frame)
        cap.release()
        # Convert frames to sequences of the correct length
        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            video_sequences.append(sequence)
    return np.array(video_sequences)