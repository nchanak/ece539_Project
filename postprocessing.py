import cv2
import numpy as np

def extract_video_sequences(video_path, sequence_length, height, width):
    cap = cv2.VideoCapture(video_path)
    raw_rgb_frames = []
    norm_rgb_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        raw_rgb_frames.append(frame_rgb)
        norm_rgb_frames.append(frame_rgb / 255.0)
    cap.release()

    def create_sequences(frames):
        return [frames[i:i+sequence_length] for i in range(len(frames) - sequence_length + 1)]

    return (
        np.array(create_sequences(raw_rgb_frames)),
        np.array(create_sequences(norm_rgb_frames))
)
