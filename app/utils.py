import tensorflow as tf
from typing import List
import cv2
import os
import numpy as np  
import json


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments


def load_recorded_video(path: str, lip_coordinates: dict, padding_percentage: float = 1.0, save_cropped: bool = True) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    sequence = []
    frame_counter = 0
    
    # Initialize the writer for the cropped video (if save_cropped is True)
    if save_cropped:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter('cropped_video.mp4', fourcc, 25.0, (140, 46))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = tf.image.rgb_to_grayscale(frame)

        # Calculate padding as a percentage of the lip width
        lip_width = lip_coordinates['x2'] - lip_coordinates['x1']
        padding = int(lip_width * padding_percentage)

        # Crop the frame around the lips with padding
        x1, y1, x2, y2 = max(lip_coordinates['x1'] - padding, 0), max(lip_coordinates['y1'] - padding, 0), \
                         lip_coordinates['x2'] + padding, lip_coordinates['y2'] + padding
        cropped_frame = gray_frame[y1:y2, x1:x2, :]

        # Resize the cropped frame to maintain aspect ratio
        # Ensure that the resized dimensions are less than or equal to 46x140
        original_height, original_width = cropped_frame.shape[:2]
        aspect_ratio = original_width / original_height
        if aspect_ratio > 140 / 46:
            new_width = 140
            new_height = int(140 / aspect_ratio)
        else:
            new_height = 46
            new_width = int(46 * aspect_ratio)

        resized_frame = cv2.resize(cropped_frame.numpy(), (new_width, new_height))

        # Pad the frame to get to 46x140
        top_padding = (46 - new_height) // 2
        bottom_padding = 46 - new_height - top_padding
        left_padding = (140 - new_width) // 2
        right_padding = 140 - new_width - left_padding
        padded_frame = cv2.copyMakeBorder(resized_frame, top_padding, bottom_padding, left_padding, right_padding, 
                                          cv2.BORDER_CONSTANT, value=0)

        # Write the frame to the output video (if save_cropped is True)
        if save_cropped:
            out.write(cv2.cvtColor(padded_frame, cv2.COLOR_GRAY2BGR))

        # Append the padded frame to the current sequence
        sequence.append(padded_frame)
        frame_counter += 1

        # If we have collected 75 frames, append the sequence to the list of sequences
        if frame_counter == 75:
            frames.append(sequence)
            sequence = []
            frame_counter = 0

    cap.release()
    if save_cropped:
        out.release()
    
    # Convert the list of sequences to a tensor and normalize it
    frames = tf.convert_to_tensor(frames, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(frames)
    normalized_frames = (frames - mean) / std

    return normalized_frames


def load_data_2(path: str, lip_coordinates: dict):
    # Load the video data
    frames = load_recorded_video(path, lip_coordinates)
    return frames







