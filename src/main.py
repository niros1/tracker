#%% 1
import argparse
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
from torchvision.ops import box_convert

from common import convert_ndarray, generate_frames, write_frame
from model import Stack, TrackinfVideoData, VideoProperties, TrackingFrameData

h264 = cv2.VideoWriter_fourcc('h','2','6','4')
mp4v = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
mp4v_2 = cv2.VideoWriter_fourcc(*'MP4V')

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
mp4v_fourcc = cv2.VideoWriter_fourcc(*'MP4V')


def extract_video_info(video_file_path) -> VideoProperties:
    try:
        cap = cv2.VideoCapture(video_file_path)

        video_properties = VideoProperties(
            fps=cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fourcc=str(cap.get(cv2.CAP_PROP_FOURCC)),
        )
        return video_properties
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()

    return None
    

def get_tracking_data(model, input_video_path, vid_props: VideoProperties, frames_limit=120) -> TrackinfVideoData:
    frame_iterator = iter(generate_frames(video_file=input_video_path, frames_limit=frames_limit))
    counter = 0
    previouse_state = {}
    # history = Stack(15)
    vid_data = TrackinfVideoData()
    for frame in tqdm(frame_iterator, total=5):
        # print(frame)
        # print(f'\rCount: {counter}', end='', flush=True)
        
        transformed_array = convert_ndarray(frame)
        boxes, logits, phrases = predict(
            model=model,
            image=transformed_array,
            caption="basketball",
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        cboxes = boxes * torch.Tensor([vid_props.width, vid_props.height, vid_props.width, vid_props.height])
        # Can't detect any object
        if boxes.shape[0] == 0:
            vid_data.all.append(
                TrackingFrameData(
                    index=counter,
                    source_index=previouse_state["counter"],
                    boxes=previouse_state["boxes"], 
                    logits=previouse_state["logits"],
                    phrases=previouse_state["phrases"],
                    cordinates=previouse_state["cordinates"],
                )
            )
            counter += 1
            continue

        xyxy_cord = box_convert(boxes=cboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xyxy_cord[0]
        t_data = TrackingFrameData(
            index=counter,
            source_index=counter,
            boxes= boxes[0].unsqueeze(0),
            logits=logits[0].unsqueeze(0),
            phrases=[phrases[0]],
            cordinates=xyxy_cord[0]
        )
        vid_data.all.append(t_data)
        previouse_state = {
            "counter": counter,
            "boxes": t_data.boxes,
            "logits": t_data.logits,
            "phrases": t_data.phrases,
            "cordinates": t_data.cordinates
        }
        counter += 1
    return vid_data

def smooth_data(data, window_size=3, use_median=False):
    # Pad the data at the start and end so we can calculate the moving average/median at the edges
    padding = window_size // 2
    data = np.pad(data, (padding, padding), 'edge')

    smoothed_data = []
    for i in range(padding, len(data) - padding):
        window = data[i - padding : i + padding + 1]
        if use_median:
            smoothed_value = np.median(window)
        else:
            smoothed_value = np.mean(window)
        smoothed_data.append(smoothed_value)

    return np.array(smoothed_data)

def main(args):
    imput_video_path = "input/basketball.mp4"
    output_video_path = "output/new_video1.mp4"
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    vid_props = extract_video_info(imput_video_path)
    frames_limit = 120


    try:
        os.remove(output_video_path)
    except:
        pass

    new_video = cv2.VideoWriter(output_video_path, mp4v_fourcc, vid_props.fps, (vid_props.width, vid_props.height))
    if not new_video.isOpened():
        print("Error: Could not open output video.")
        exit()

    if args.force_create_tracking or not os.path.exists("output/tracking_data.pkl"):
        tracking_data = get_tracking_data(model, imput_video_path, vid_props, frames_limit=frames_limit)
        pickle.dump(tracking_data, open("output/tracking_data.pkl", "wb"))
    else:
        tracking_data = pickle.load(open("output/tracking_data.pkl", "rb"))
    
    X = [c.cordinates[0] for c in tracking_data.all]
    tracking_data.X = smooth_data(X, window_size=100, use_median=True)
    Y = [c.cordinates[1] for c in tracking_data.all]
    tracking_data.Y = smooth_data(Y, window_size=100, use_median=True)


    # print(tracking_data)
    frame_iterator = iter(generate_frames(video_file=imput_video_path, frames_limit=frames_limit))
    # frames_data = []
    counter = 0
    # previouse_state = {}
    history = Stack(30)
    for frame in tqdm(frame_iterator, total=5):
        track_data = tracking_data.all[counter]
        doc_str = write_frame(new_video, frame, history,  track_data,tracking_data.X[counter], tracking_data.Y[counter],track_data.logits, track_data.phrases)
        history.push(doc_str)
        counter += 1
 
    print("Releasing video")
    new_video.release()


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLFlow console"
    )
    parser.add_argument(
        "--force-create-tracking",
        action="store_true",
        default=os.getenv(
            "FORCE_CREATE_TRACK_DATA",
            False,
        ),
    )
    args = parser.parse_args()
    main(args)
