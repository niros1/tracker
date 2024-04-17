# %% 1
import argparse
from ast import arg
import os
import pickle
import sys
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import time
import torch
from torchvision.ops import box_convert
from loguru import logger

from common import (
    convert_ndarray,
    extract_audio_from_frames,
    retrieve_frames,
    write_frame,
    attach_audio,
)
from model import Stack, TrackinfVideoData, VideoProperties, TrackingFrameData

h264 = cv2.VideoWriter_fourcc("h", "2", "6", "4")
mp4v = cv2.VideoWriter_fourcc("m", "p", "4", "v")
mp4v_2 = cv2.VideoWriter_fourcc(*"MP4V")
xvid_fourcc = cv2.VideoWriter_fourcc(*"XVID")
mp4v_fourcc = cv2.VideoWriter_fourcc(*"MP4V")


BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

FOURCC = mp4v


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
                    source_index=previouse_state["counter"]
                    if "counter" in previouse_state
                    else counter,
                    boxes=previouse_state["boxes"]
                    if "boxes" in previouse_state
                    else torch.Tensor([]),
                    logits=previouse_state["logits"]
                    if "logits" in previouse_state
                    else torch.Tensor([]),
                    phrases=previouse_state["phrases"]
                    if "phrases" in previouse_state
                    else [],
                    cordinates=previouse_state["cordinates"]
                    if "cordinates" in previouse_state
                    else None,
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
                    source_index=previouse_state["counter"]
                    if "counter" in previouse_state
                    else counter,
                    boxes=previouse_state["boxes"]
                    if "boxes" in previouse_state
                    else torch.Tensor([]),
                    logits=previouse_state["logits"]
                    if "logits" in previouse_state
                    else torch.Tensor([]),
                    phrases=previouse_state["phrases"]
                    if "phrases" in previouse_state
                    else [],
                    cordinates=previouse_state["cordinates"]
                    if "cordinates" in previouse_state
                    else None,
                )
            )
            counter += 1
            continue

        # Convert the boxes to the frame size
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
        folder_path = args.process_folder
    if args.process_file != "":
        files = args.process_file
    for file_name in files:
        file_path = f"{folder_path}/{file_name}"
        model = load_gd_model()
        # from groundingdino.util.inference import load_model, load_image, predict, annotate

        vid_props = extract_video_info(file_path)

        file_name_no_ext = os.path.splitext(file_name)[0]
        os.makedirs(f"{folder_path}/tracking", exist_ok=True)

        # game_name = os.path.basename(os.path.dirname(file_path))
        # dir_path = os.path.dirname(file_path)
        pickle_name = f"{folder_path}/tracking/tracking_data_{file_name_no_ext}.pkl"
        logger.info(
            f"Creating tracking data for {args.force_create_tracking} - {pickle_name}"
        )

        if args.force_create_tracking or not os.path.exists(pickle_name):
            tracking_data = get_tracking_data(
                model, file_path, vid_props, frames_limit=args.tracking_frames_limit
            )
            pickle.dump(tracking_data, open(pickle_name, "wb"))

        if args.create_video is True:
            tracking_data = pickle.load(open(pickle_name, "rb"))
            process_video(
                args, file_path, model, vid_props, tracking_data, args.attach_sound
            )


def load_gd_model():
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
    except Exception as e:
        print("groundingdino module is not available, traking data can't be processed")
        model = None
    return model


def process_video(
    args,
    file_path,
    model,
    vid_props,
    tracking_data: TrackinfVideoData,
    attach_sound=False,
):
    """Construct a new video with the tracking data

    Args:
        args (_type_): _description_
        file_path (_type_): _description_
        model (_type_): _description_
        vid_props (_type_): _description_
        tracking_data (TrackinfVideoData): _description_
    """
    # vid_props = extract_video_info(file_path)
    # tracking_frames_limit = args.tracking_frames_limit
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    game_name = os.path.basename(os.path.dirname(file_path))
    window_size = 300

    # Traking data manipulation
    X = [c.cordinate[0] for c in tracking_data.all if c.cordinate is not None]
    tracking_data.X = smooth_data(X, window_size=window_size, use_median=False)
    tracking_data.X = smooth_data(
        tracking_data.X, window_size=window_size, use_median=True
    )
    Y = [c.cordinate[1] for c in tracking_data.all if c.cordinate is not None]
    tracking_data.Y = smooth_data(Y, window_size=window_size, use_median=False)

    plot_smoothing_curve(tracking_data, file_name_no_ext, X, tracking_data.X)

    # Create output video
    out_vid_len_frames = args.out_vid_len
    starting_point = args.start_frame
    os.makedirs(f"output/{game_name}", exist_ok=True)
    # output_video_path = (
    #     f"output/{game_name}/video_{file_name_no_ext}_{out_vid_len_frames}.mp4"
    # )
    os.makedirs(f"output/{game_name}", exist_ok=True)
    output_video_path = (
        f"output/{game_name}/video_{file_name_no_ext}_{out_vid_len_frames}.mp4"
    )
    output_video_path_aud = (
        f"output/video_{file_name_no_ext}_{out_vid_len_frames}_aud.mp4"
    )
    output_video = get_video_writer(output_video_path, vid_props)

    frame_iterator = iter(
        retrieve_frames(
            video_file=file_path,
            frames_limit=out_vid_len_frames,
            starting_point=starting_point,
        )
    )
    frame_index = starting_point
    history = Stack(30)
    for frame in tqdm(frame_iterator, total=5):
        track_data: TrackingFrameData = tracking_data.all[frame_index]
        start_time = time.time()  # Start timing
        history.push(str(track_data))
        doc_str = write_frame(
            output_video,
            frame,
            history,
            track_data,
            tracking_data.X[frame_index],
            tracking_data.Y[frame_index],
            track_data.logits,
            track_data.phrases,
            draw_blind_spots=True,
            draw_tracking=False,
            write_history=False,
        )
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        # logger.info(f"Time elapsed for this iteration: {elapsed_time} seconds")

        frame_index += 1

    print(f"Releasing video {output_video_path}")
    output_video.release()

    if attach_sound:
        output_video_path_aud = attach_audio_to_video(
            file_path, output_video_path, file_name_no_ext
        )

        # extract_audio_from_frames(file_path, starting_point, out_vid_len_frames)
        # attach_audio(file_path, output_video_path, output_video_path_aud)


def attach_audio_to_video(audio_vid_path, destinamtion_video_path, aac_file_name):
    import subprocess

    print(f"Attaching audio to {destinamtion_video_path}")
    command = (
        f"ffmpeg -y -i {audio_vid_path} -vn -acodec copy output/{aac_file_name}.aac"
    )
    subprocess.run(command, shell=True, check=True)
    output_video_path_aud = destinamtion_video_path.replace(".mp4", "_aud.mp4")
    command = f"ffmpeg -y -i {destinamtion_video_path} -i output/{aac_file_name}.aac -c:v copy -c:a aac {output_video_path_aud}"
    subprocess.run(command, shell=True, check=True)
    print(f"Audio attached to {output_video_path_aud}")
    return output_video_path_aud


def plot_smoothing_curve(tracking_data, file_name_no_ext, X, Y):
    size = 5000
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(X))[:size], np.array(X)[:size], label="Original X")
    plt.plot(
        np.arange(len(tracking_data.X))[:size], tracking_data.X[:size], label="Smooth X"
    )
    plt.title("X")
    # plt.legend()
    # plt.savefig(f"output/plot_x_{file_name_no_ext}.png")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(Y)), np.array(Y), label="Original Y")
    plt.plot(np.arange(len(tracking_data.Y)), tracking_data.Y, label="Smooth Y")
    plt.title("Y")

    plt.legend()
    plt.savefig(f"output/plot_xy_{file_name_no_ext}.png")


def get_video_writer(output_video_path, vid_props, fourcc=FOURCC):
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


def set_args():
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
        "--attach-sound",
        action="store_true",
        default=os.getenv(
            "ATTACH_SOUND",
            True,
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
        nargs="*",
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
            "OUT_VID_LEN",
            1000,
        ),
    )

    parser.add_argument(
        "--start-frame",
        action="store",
        type=int,
        default=os.getenv(
            "START_FRAME",
            0,
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
    return args


# %%
if __name__ == "__main__":
    logger.remove()

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    logger.add(
        sys.stdout,
        format="{time} {level} {message}",
        # filter="gradio-server",
        level="INFO",
    )

    args = set_args()
    main(args)
