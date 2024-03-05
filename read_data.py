import os
import warnings

import cv2
import imageio


def read_video_to_frames(video_file):

    if not os.path.exists(video_file):
        warnings.warn("file does not exist.", UserWarning)
        return None

    if video_file.endswith(".mov"):
        return read_mov_data_to_frames(video_file)
    else:
        return read_mp4_data_to_frames(video_file)


def read_mp4_data_to_frames(video_file):
    video = cv2.VideoCapture(video_file)
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def read_mov_data_to_frames(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    for image in vid.iter_data():
        frames.append(image)
    return frames