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
        # Read each frame from the video, resizing, and normalizing
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame = frame / 255.0
            frames.append(frame)
        cap.release()

        # Creates a list of frame sequences of sequence_length length, where each sequence is a list of frames
        # They overlap by sequence_length - 1 frames
        # This means that if sequence_length is 10, the first sequence will be frames[0:10], the second will be frames[1:11], etc.
        for i in range(len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            # Add the sequence to the list of sequences, which contains the sequences for ALL videos
            video_sequences.append(sequence)
            # This mapping keeps track of what video a sequence is from via index, kinda deprecated
            sequence_filenames.append(video_file)

    return np.array(video_sequences), sequence_filenames
