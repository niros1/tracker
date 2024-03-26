from ast import Tuple
import os
import cv2
from scipy.fftpack import sc_diff
import matplotlib.pyplot as plt
import cv2 as cv
from typing import Any, Generator
import numpy as np
import torch
from PIL import Image

from torchvision.ops import box_convert
from model import Stack, blind_spots, TrackingFrameData
from moviepy.editor import VideoFileClip


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def extract_frames(video_path, output_folder, frames_limit=100, skip=0):
    """
    write each frame to a file
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Could not open video")
        return

    frame_taken = 0
    iteration = -1
    success = True
    files = []
    while (frames_limit > 0 and frame_taken < frames_limit) or (
        frames_limit == 0 and success is True
    ):
        iteration += 1

        # Read the next frame from the video. If you read at the end of the video, success will be False
        success, frame = video.read()
        # print(frame_count)

        # Break the loop if the video is finished
        if not success:
            break
        if skip != 0 and iteration % skip != 0:
            continue

        # Save the frame into the output folder
        cv2.imwrite(f"{output_folder}/frame{frame_taken}.jpg", frame)
        files.append(f"{output_folder}/frame{frame_taken}.jpg")

        frame_taken += 1

    # Release the video file
    video.release()
    return files


def retrieve_frames(
    video_file: str, frames_limit=10, starting_point=0, accuracy=0
) -> Generator[np.ndarray, None, None]:
    import time

    """
        yield each frame as byte array
    """
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"No such file: '{video_file}'")
    video = cv2.VideoCapture(video_file)

    frame_count = 0
    # frames_limit += starting_point
    start_time = time.time()
    print(f"Starting Point: {starting_point} frames")
    video.set(cv2.CAP_PROP_POS_FRAMES, starting_point)
    while video.isOpened():
        # if frame_count < starting_point:
        #     print(f"\rSkipping {starting_point} frames ", end="")
        #     # frame_count += 1
        #     video.grab()
        #     continue

        # Skip frame (performance optimizztion)
        if accuracy != 0 and frame_count % accuracy != 0:
            # print("before grab time:", time.time() - start_time)

            success = video.grab()
            # print("grab time:", time.time() - start_time)
            frame = None
        else:
            # print("before read time:", time.time() - start_time)

            success, frame = video.read()
            # print("read time:", time.time() - start_time, frame)

        if (frames_limit > 0 and frame_count >= frames_limit) or (
            frames_limit == 0 and success is False
        ):
            break

        yield frame
        frame_count += 1

    video.release()


def plot_image(image: np.ndarray, size: int = 12) -> None:
    # %matplotlib inline
    plt.figure(figsize=(size, size))
    plt.imshow(image[..., ::-1])
    plt.show()


def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def convert_ndarray(frame: np.ndarray[Any]) -> torch.Tensor:
    import groundingdino.datasets.transforms as T

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_source = Image.fromarray(frame_rgb)

    # image_source = Image.fromarray(arr).convert("RGB")
    # image_source = Image.open(image_path).convert("RGB")
    # image_source = Image.open(image_path).convert("RGB")
    # image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def add_text_to_frame2(
    frame, text, position=(50, 50), font_scale=1, font_color=(0, 0, 255), thickness=4
):
    """
    Adds text to a single frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if is_iterable(text):
        for line in text:
            cv2.putText(
                frame, f"{line}", position, font, font_scale, font_color, thickness
            )
            position = (position[0], position[1] + 30)
    else:
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    return frame


def write_frame(
    vid_writer,
    source_frame,
    history: Stack,
    tracking_data: TrackingFrameData,
    x,
    y,
    logits,
    phrases,
):
    # annotated_frame = annotate(image_source=source_frame, boxes=tracking_data.boxes, logits=logits, phrases=phrases)

    # y = 800
    # x = 500
    frame_with_bbox = source_frame.copy()

    # Blind spots bounding boxes
    # frame_with_bbox = draw_bounding_boxes(
    #     source_frame, blind_spots, tracking_data.phrases
    # )

    anotations = tracking_data.cordinates
    anotations = [
        [(int(x), int(y)) for x, y in zip(arr[::2], arr[1::2])] for arr in anotations
    ]

    # Tracking bounding boxes
    # frame_with_bbox = draw_bounding_boxes(
    #     frame_with_bbox, anotations, tracking_data.phrases, color=(0, 255, 0)
    # )

    zoom_frame = zoom_at(frame_with_bbox, 2, coord=(x, y))

    # print(f"Frame {tracking_data.index}->>>>>", (x, y))

    # add_text_to_frame2(zoom_frame, history, position=(50, 150))
    vid_writer.write(zoom_frame)
    return f"source idx{tracking_data.source_index}  idx {tracking_data.index}"


def draw_bounding_boxes(image, boxes, labels, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    """
    # for box, label in zip(boxes, labels):
    for box in boxes:
        # x1, y1, x2, y2 = box
        left_upper, right_lower = box
        cv2.rectangle(image, left_upper, right_lower, color, thickness)
        # cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def attach_audio(
    org_vid_path, output_vid_path, output_video_path_aud, audio_subclip=None
):
    """This one will attach the whole audio from the original video to the output video.

    Args:
        org_vid_path (_type_): _description_
        output_vid_path (_type_): _description_
        output_video_path_aud (_type_): _description_
    """

    if audio_subclip is None:
        # Load the original video and get its audio
        original_video = VideoFileClip(org_vid_path)
        original_audio = original_video.audio
    else:
        original_audio = audio_subclip

    # Load the output video
    output_video = VideoFileClip(output_vid_path)

    # Add the audio to the output video
    final_video = output_video.set_audio(original_audio)

    # Write the final video to a file
    final_video.write_videofile(output_video_path_aud)


def extract_audio_from_frames(org_vid_path, start_frame, end_frame):
    """This one will extract audio from the specified frames.

    Args:
        org_vid_path (_type_): _description_
        start_frame (_type_): _description_
        end_frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load the original video and get its audio
    original_video = VideoFileClip(org_vid_path)
    original_audio = original_video.audio

    # Calculate start and end times in seconds
    fps = original_video.fps  # frames per second
    start_time = start_frame / fps
    end_time = end_frame / fps

    # Extract audio from the specified frames
    audio_subclip = original_audio.subclip(start_time, end_time)

    return audio_subclip
