from ast import Tuple
import cv2
from scipy.fftpack import sc_diff
import matplotlib.pyplot as plt
import cv2 as cv
from typing import Any, Generator
import numpy as np
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
from model import Stack, TrackingFrameData

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
    while (frames_limit > 0 and frame_taken < frames_limit) or (frames_limit == 0 and success is True):
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

        frame_taken +=1

    # Release the video file
    video.release()
    return files

def generate_frames(video_file: str, frames_limit=10) -> Generator[np.ndarray, None, None]:
    """
        yield each frame as byte array
    """
    video = cv2.VideoCapture(video_file)
    frame_count = 0

    while video.isOpened():
        success, frame = video.read()

        if not ((frames_limit > 0 and frame_count < frames_limit) or (frames_limit == 0 and success is True)):
            break

        yield frame
        frame_count += 1

    video.release()

def plot_image(image: np.ndarray, size: int = 12) -> None:
    # %matplotlib inline
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show()

def zoom_at(img, zoom=1, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result

def convert_ndarray(frame: np.ndarray[Any]) ->  torch.Tensor:
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

def add_text_to_frame2(frame, text, position=(50, 50), font_scale=1, font_color=(0, 0, 255), thickness=4):
    """
    Adds text to a single frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if is_iterable(text):
        for line in text:
            cv2.putText(frame, f"{line}", position, font, font_scale, font_color, thickness)
            position = (position[0], position[1] + 30)
    else:
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness)
    return frame

def write_frame(vid_writer, source_frame, history: Stack,  tracking_data: TrackingFrameData, x, y,  logits, phrases):
    # annotated_frame = annotate(image_source=source_frame, boxes=tracking_data.boxes, logits=logits, phrases=phrases)

    # y = 800
    # x = 500

    zoom_frame = zoom_at(source_frame, 2, coord=(x, y))

    print(f"Frame {tracking_data.index}->>>>>", (x, y))
    
    add_text_to_frame2(zoom_frame, history, position=(50, 150))
    vid_writer.write(zoom_frame)
    return f"Frame {tracking_data.index} - Zoom at: {x}, {y} ---- phrases: {phrases}"