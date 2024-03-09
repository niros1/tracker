# %% 1
import argparse
from ast import arg
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch
from torchvision.ops import box_convert

from common import convert_ndarray, retrieve_frames, write_frame
from model import Stack, TrackinfVideoData, VideoProperties, TrackingFrameData

h264 = cv2.VideoWriter_fourcc("h", "2", "6", "4")
mp4v = cv2.VideoWriter_fourcc("m", "p", "4", "v")
mp4v_2 = cv2.VideoWriter_fourcc(*"MP4V")

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
mp4v_fourcc = cv2.VideoWriter_fourcc(*"MP4V")


def extract_video_info(video_file_path) -> VideoProperties | None:
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


def get_tracking_data(
    model, input_video_path, vid_props: VideoProperties, frames_limit=120, accuracy=40
) -> TrackinfVideoData:
    from groundingdino.util.inference import predict

    print(
        f"Getting tracking data for {input_video_path} with limit {frames_limit} accuracy {accuracy}"
    )
    frame_iterator = iter(
        retrieve_frames(
            video_file=input_video_path, frames_limit=frames_limit, accuracy=accuracy
        )
    )
    counter = 0
    previouse_state = {}
    # history = Stack(15)
    vid_data = TrackinfVideoData()
    for frame in tqdm(
        frame_iterator,
        desc="get_tracking_data",
        total=frames_limit if frames_limit > 0 else None,
    ):
        if counter % accuracy != 0:
            # Use the data of the last frame which was processed
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

        transformed_array = convert_ndarray(frame)
        boxes, logits, phrases = predict(
            model=model,
            image=transformed_array,
            caption="basketball",
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )

        # Can't detect any object, use the previous frame data
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

        cboxes = boxes * torch.Tensor(
            [vid_props.width, vid_props.height, vid_props.width, vid_props.height]
        )
        xyxy_cord = box_convert(boxes=cboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        t_data = TrackingFrameData(
            index=counter,
            source_index=counter,
            # boxes=boxes[0].unsqueeze(0),
            # logits=logits[0].unsqueeze(0),
            # phrases=[phrases[0]],
            # cordinates=xyxy_cord[0],
            boxes=boxes,
            logits=logits,
            phrases=phrases,
            cordinates=xyxy_cord,
        )
        vid_data.all.append(t_data)
        previouse_state = {
            "counter": counter,
            "boxes": t_data.boxes,
            "logits": t_data.logits,
            "phrases": t_data.phrases,
            "cordinates": t_data.cordinates,
        }
        counter += 1
    return vid_data


def smooth_data(data, window_size=3, use_median=False):
    # Pad the data at the start and end so we can calculate the moving average/median at the edges
    padding = window_size // 2
    data = np.pad(data, (padding, padding), "edge")

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
    if args.process_folder == "" and args.process_file == "":
        print("No file or folder to process")
        exit()

    if args.process_folder != "":
        folder_name = args.process_folder
    if args.process_file != "":
        file_path = args.process_file

    try:
        from groundingdino.util.inference import (
            load_model,
            load_image,
            predict,
            annotate,
        )

        model = load_model(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth",
        )
    except ImportError:
        print("groundingdino module is not available, traking data can't be processed")
        model = None
    # from groundingdino.util.inference import load_model, load_image, predict, annotate

    vid_props = extract_video_info(file_path)

    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    pickle_name = f"output/tracking_data_{file_name_no_ext}.pkl"

    if args.force_create_tracking or not os.path.exists(pickle_name):
        tracking_data = get_tracking_data(
            model, file_path, vid_props, frames_limit=args.tracking_frames_limit
        )
        pickle.dump(tracking_data, open(pickle_name, "wb"))

    if args.create_video is True:
        tracking_data = pickle.load(open(pickle_name, "rb"))
        process_video(args, file_path, model, vid_props, tracking_data)


def process_video(args, file_path, model, vid_props, tracking_data: TrackinfVideoData):
    # vid_props = extract_video_info(file_path)
    # tracking_frames_limit = args.tracking_frames_limit
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    # Traking data manipulation
    X = [c.cordinate[0] for c in tracking_data.all]
    tracking_data.X = smooth_data(X, window_size=100, use_median=True)
    Y = [c.cordinate[1] for c in tracking_data.all]
    tracking_data.Y = smooth_data(Y, window_size=100, use_median=True)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(X)), np.array(X), label="Original X")
    plt.plot(np.arange(len(tracking_data.X)), tracking_data.X, label="Smooth X")
    plt.title("X")
    # plt.legend()
    # plt.savefig(f"output/plot_x_{file_name_no_ext}.png")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(Y)), np.array(Y), label="Original Y")
    plt.plot(np.arange(len(tracking_data.Y)), tracking_data.Y, label="Smooth Y")
    plt.title("Y")

    plt.legend()
    plt.savefig(f"output/plot_xy_{file_name_no_ext}.png")
    exit()
    # Create output video
    out_vid_len_frames = args.out_vid_len
    output_video_path = f"output/video_{file_name_no_ext}_{out_vid_len_frames}.mp4"
    output_video = get_video_writer(output_video_path, vid_props)

    frame_iterator = iter(
        retrieve_frames(video_file=file_path, frames_limit=out_vid_len_frames)
    )
    counter = 0
    history = Stack(30)
    for frame in tqdm(frame_iterator, total=5):
        track_data = tracking_data.all[counter]
        doc_str = write_frame(
            output_video,
            frame,
            history,
            track_data,
            tracking_data.X[counter],
            tracking_data.Y[counter],
            track_data.logits,
            track_data.phrases,
        )
        history.push(doc_str)
        counter += 1

    print(f"Releasing video {output_video_path}")
    output_video.release()


def get_video_writer(output_video_path, vid_props, fourcc=mp4v_fourcc):
    try:
        os.remove(output_video_path)
    except:
        pass
    output_video = cv2.VideoWriter(
        output_video_path, fourcc, vid_props.fps, (vid_props.width, vid_props.height)
    )
    if not output_video.isOpened():
        print("Error: Could not open output video.")
        exit()
    return output_video


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLFlow console")
    parser.add_argument(
        "--force-create-tracking",
        action="store_true",
        default=os.getenv(
            "FORCE_CREATE_TRACK_DATA",
            False,
        ),
    )

    parser.add_argument(
        "--create-video",
        action="store_true",
        default=os.getenv(
            "CREATE_VIDEO",
            False,
        ),
    )

    parser.add_argument(
        "--process-folder",
        action="store",
        default=os.getenv(
            "PROCESSES_FOLDER",
            "",
        ),
    )

    parser.add_argument(
        "--process-file",
        action="store",
        default=os.getenv(
            "PROCESSES_FILE",
            "",
        ),
    )

    parser.add_argument(
        "--out-vid-len",
        action="store",
        type=int,
        default=os.getenv(
            "PROCESSES_FILE",
            1000,
        ),
    )

    parser.add_argument(
        "--tracking-frames-limit",
        action="store",
        type=int,
        default=os.getenv(
            "PROCESSES_FILE",
            1200,  # 0 means no limit
        ),
    )

    args = parser.parse_args()

    print(args)
    main(args)
