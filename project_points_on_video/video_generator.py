import cv2
import numpy as np


def avi_video():
    # Set video parameters
    width, height = (1280, 1024)
    fps = 25
    duration_seconds = 60
    num_frames = int(fps * duration_seconds)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    # Create black frames and write them to the video
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), np.uint8)  # Black frame
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video 'output.avi' created successfully with {num_frames} black frames.")


def mp4_video():
    # Set video parameters
    width, height = (1280, 1024)
    fps = 25
    duration_seconds = 60
    num_frames = int(fps * duration_seconds)

    # Define the codec and create a VideoWriter object for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Create black frames and write them to the video
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), np.uint8)  # Black frame
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Video 'output.mp4' created successfully with {num_frames} black frames.")


if __name__ == "__main__":
    # avi_video()
    mp4_video()
