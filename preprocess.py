import os
import cv2
import numpy as np

def preprocess_videos_with_mapping(video_folder, sequence_length, height, width, channels):
    video_sequences = []
    sequence_filenames = []

    for video_file in sorted(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame = frame / 255.0
            frames.append(frame)
        cap.release()

        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            video_sequences.append(sequence)
            sequence_filenames.append(video_file)

    return np.array(video_sequences), sequence_filenames
