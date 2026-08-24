import argparse
import os
import pickle
import sys
import tkinter as tk
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import face_recognition
import numpy as np
import torch
from moviepy import VideoFileClip
from tqdm import tqdm

from model.face_detection import FaceDetector
from model.head_pose import HeadPose
from model.gaze_estimation.gaze import GazeEstimator
from model.fer import FER
from model.face_parse import FaceParser
from model.physio import Physio
from model.whisper_wrapper import WhisperWrapper

from deep_sort_realtime.deepsort_tracker import DeepSort

from collections import defaultdict
from statistics import mean


def safe_videoclip(path, retries=3, delay=0.5, **kwargs):
    for attempt in range(1, retries+1):
        try:
            return VideoFileClip(path, **kwargs)
        except OSError as e:
            print(f"[WARN] Attempt {attempt} failed for {os.path.basename(path)}: {e}")
            time.sleep(delay)
    raise OSError(f"Failed to load {path} after {retries} attempts")

class FaceTracker(object):
    def __init__(self):
        self.head_pose = HeadPose()
        
        self.gaze_estimator = GazeEstimator("resnet34")
        
        self.fer = FER(use_audio=False)
        
        self.face_parser = FaceParser()
        
        self.physio = Physio()
        
        self.whisper_wrapper = WhisperWrapper()
        
        self.tracker = DeepSort(max_age=2, n_init=2)

        # JPEG crops are encoded+written to disk in the background so that
        # I/O doesn't stall the GPU-bound inference loop. Futures are tracked
        # so we can wait for them (and surface any write errors) at the end
        # of each video, instead of leaving an unbounded backlog.
        self.io_executor = ThreadPoolExecutor(max_workers=2)
        self.pending_saves = []

        self.reset()

    def reset(self):
        self.frame_count = 0
        self.face_trackers = []
        self.known_face_encodings = []
        self.known_face_ids = []
        self.next_face_id = 1
        self.data_to_save = {}

        self.tracker.delete_all_tracks()

        if hasattr(self, "physio"):
            self.physio.reset_image_queue()

    def save_image_async(self, img_pil, path):
        self.pending_saves.append(self.io_executor.submit(img_pil.save, path))

    def wait_for_pending_saves(self):
        if not self.pending_saves:
            return
        for future in self.pending_saves:
            future.result()  # re-raise any exception from the background save
        self.pending_saves.clear()

    def shutdown(self):
        self.wait_for_pending_saves()
        self.io_executor.shutdown(wait=True)

    def export_data(self, out_path, movie_name):
        with open(os.path.join(out_path, movie_name, "data.pkl"), "wb") as f:
            pickle.dump(self.data_to_save, f)

    def compute_face(self, frame=None, frame_batch=None):
        # boxes = face_detector.inference(input_frame=frame)
        boxes_list = face_detector.inference_batch(input_frame_batch=frame_batch)

        bbs_list = []
        for boxes in boxes_list:
            bbs = [([x,y,w,h],0.7,"face") for (x,y,w,h) in boxes]
            
            bbs_list.append(bbs)
        
        return bbs_list
        
    def compute_trackers(self, bbs, frames):
        output_bbs, output_face_ids = [], []

        step = TRACKER_STEP

        # tracker.update_tracks() runs an expensive Re-ID embedder (MobileNet)
        # forward pass whenever there are detections, and is by far the
        # dominant cost of feature extraction. The "end" of one step-sized
        # window is the exact same frame as the "start" of the next window
        # (bbs[i+step] here == bbs[start] next iteration), so carry that
        # result forward instead of calling update_tracks a second time on
        # the identical input -- this was roughly doubling the number of
        # (expensive) embedder calls for no benefit.
        prev_end_bbs, prev_end_face_ids = None, None

        for i in range(0, len(bbs), step):
            start = i
            end = i + step

            if end >= len(bbs):
                break

            # Start
            if prev_end_bbs is not None:
                start_bbs, start_face_ids = prev_end_bbs, prev_end_face_ids
            else:
                start_bbs, start_face_ids = [], []
                self.tracks = self.tracker.update_tracks(bbs[start], frame=frames[start])

                for tracker in self.tracks:
                    if not tracker.is_confirmed():
                        continue

                    face_id = tracker.track_id
                    box = [int(p) for p in tracker.to_ltrb()]

                    start_bbs.append(box)
                    start_face_ids.append(face_id)

            # End
            end_dict = {}
            self.tracks = self.tracker.update_tracks(bbs[end], frame=frames[end])

            for tracker in self.tracks:
                if not tracker.is_confirmed():
                    continue

                face_id = tracker.track_id
                box = [int(p) for p in tracker.to_ltrb()]

                end_dict[face_id] = box

            output_bbs.append(start_bbs)
            output_face_ids.append(start_face_ids)
            for i in range(step-2):
                output_bbs.append([])
                output_face_ids.append([])

            for start_bb, face_id in zip(start_bbs, start_face_ids):
                if face_id in end_dict:
                    end_bb = end_dict[face_id]

                    interpolated_bbs = np.linspace(start=start_bb, stop=end_bb, num=step)[1:-1]
                    for i,interpolated_bb in enumerate(interpolated_bbs):
                        output_bbs[-step+2+i].append(interpolated_bb.tolist())
                        output_face_ids[-step+2+i].append(face_id)

            output_bbs.append([box for face_id,box in end_dict.items()])
            output_face_ids.append([face_id for face_id,box in end_dict.items()])

            prev_end_bbs, prev_end_face_ids = output_bbs[-1], output_face_ids[-1]

        return output_bbs, output_face_ids
            

    def update_face_trackers(self, frame, output_bbs, output_face_ids, movie_name, out_path):
        if not output_face_ids:
            return

        # The visualization window is disabled (see main loop, cv2.imshow is
        # commented out), so we no longer draw boxes/labels onto `frame`. That
        # means we don't need a defensive per-face frame.copy() anymore (it was
        # also a correctness bug: since it was taken *after* an earlier face's
        # rectangle had already been drawn onto the shared `frame`, a second
        # face in the same frame could get a "clean" copy that actually
        # contained the first face's annotation). It also means the BGR->RGB
        # conversion is identical for every face in this frame and can be
        # computed once and shared, instead of redone per face per model.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for face_id, box in zip(output_face_ids, output_bbs):
            box = [int(p) for p in box]
            x1, y1, x2, y2 = box

            # Physiological measurements
            if hasattr(self, "physio"):
                self.physio.add_image_to_queue(frame_rgb, [x1, y1, x2-x1, y2-y1], face_id)

            if not (self.frame_count > SAVE_DATA_INTERVAL and self.frame_count % SAVE_DATA_INTERVAL == 0):
                continue
            # if os.path.exists(os.path.join(out_path, f"{movie_name}_{self.frame_count}_person{face_id}.pkl")):
            #     continue

            # Obtain head pose and draw it
            if hasattr(self, "head_pose"):
                hp = self.head_pose.get_head_pose(frame_rgb, [x1, y1, x2-x1, y2-y1])
                # roll, pitch, yaw = hp
                # self.head_pose.draw_headpose(frame, hp, [x1, y1, x2-x1, y2-y1])

            # Obtain gaze and draw it
            if hasattr(self, "gaze_estimator"):
                pitch_predicted, yaw_predicted = self.gaze_estimator.inference(frame_rgb, [x1, y1, x2-x1, y2-y1])
                # self.gaze_estimator.draw_gaze(frame, [x1, y1, x2-x1, y2-y1], pitch_predicted, yaw_predicted)

            # Facial emotion
            if hasattr(self, "fer"):
                emotion_video = self.fer.inference_video(frame_rgb, [x1, y1, x2-x1, y2-y1])
                # label = f"Person {face_id}: {emotion}"
                # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Physio
            if hasattr(self, "physio"):
                rPPG = self.physio.inference(face_id)

            # Face (face_parser does its own internal BGR->RGB conversion, so it
            # still gets the original BGR frame here)
            if hasattr(self, "face_parser"):
                parsing, img_pil = self.face_parser.inference(frame, [x1, y1, x2-x1, y2-y1])
                # self.face_parser.vis_parsing_maps(img_pil, parsing, stride=1, save_im=True, save_path=os.path.join("debug", f"{movie_name}_{self.frame_count}_person{face_id}.png"))
            
                bbox_mouth = self.get_bounding_box(parsing)
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"] = {
                    "bbox_mouth": bbox_mouth
                }
                # with open(os.path.join(out_path, movie_name, f"{face_id}_{face_tracker.frame_count}.pkl"), "wb") as f:
                #     pickle.dump(bbox_mouth, f)
            
                self.save_image_async(img_pil, os.path.join(out_path, movie_name, "frames", f"{face_id}_{face_tracker.frame_count}.jpg"))
            
            # Save crop
            # crop = frame[y:y+h,x:x+w]
            # cv2.imwrite(os.path.join(out_path, f"{movie_name}_{self.frame_count}_person{face_id}.png"), crop)
            # Save to file
            if hasattr(self, "head_pose"):
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"]["hp"] = hp
            if hasattr(self, "gaze_estimator"):
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"]["gaze"] = [pitch_predicted.astype(np.float16), yaw_predicted.astype(np.float16)]
            if hasattr(self, "fer"):
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"]["emotion_video"] = emotion_video
            if hasattr(self, "physio"):
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"]["rPPG"] = rPPG,
            if hasattr(self, "face_parser"):
                self.data_to_save[f"{face_id}_{face_tracker.frame_count}"]["face_parse"] = parsing.astype(np.uint8)
        
    def get_bounding_box(self, segmentation_map, target_label=11, margin=5):
        ys, xs = np.where(segmentation_map == target_label)
        
        if len(xs) == 0 or len(ys) == 0:
            return None  # Mouth not found
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Add margin, clip to image bounds
        h, w = segmentation_map.shape
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w - 1)
        y_max = min(y_max + margin, h - 1)
        
        return (np.int16(x_min), np.int16(y_min), np.int16(x_max), np.int16(y_max))
                            

RECOMPUTE_FACES_INTERVAL = 1
SAVE_DATA_INTERVAL = 6

# Stride (in real video frames) between DeepSort tracking updates in
# compute_trackers. The main loop below only calls update_face_trackers once
# per TRACKER_STEP frames (on the actually-tracked frame, not one of the
# interpolated in-between ones), so it must match compute_trackers' own step.
TRACKER_STEP = 5

# Frames are read and processed in windows of this many frames at a time,
# instead of loading an entire video into RAM up front. Must comfortably
# exceed the tracker's step size (5) so that only a small fraction of frames
# is ever dropped at a window boundary (see compute_trackers).
FRAME_WINDOW_SIZE = 600

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def discover_video_files(input_path, extensions=VIDEO_EXTENSIONS):
    """Recursively find video files under input_path, returning absolute paths."""
    video_paths = []
    for root, _, files in os.walk(input_path):
        for file_name in files:
            if file_name.lower().endswith(extensions):
                video_paths.append(os.path.join(root, file_name))
    return sorted(video_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-frame face/audio features from a directory of videos.")
    parser.add_argument("--input_path", type=str, default="/home/eivor/data/MAVOS-DD",
                         help="Root directory to recursively search for video files.")
    parser.add_argument("--output_path", type=str, default="interim_outputs2",
                         help="Root directory where extracted features are written, mirroring input_path's structure.")
    parser.add_argument("--frame_window_size", type=int, default=FRAME_WINDOW_SIZE,
                         help="Number of frames to decode/process at a time, instead of loading a whole video into RAM.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    FRAME_WINDOW_SIZE = args.frame_window_size

    # Create a named window that we can resize and move
    window_name = "Multi-Face Tracking + Recognition"
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # Move the window
    def get_screen_size():
        # Get screen size using tkinter
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Calculate top-left corner for centering the window
        window_width, window_height = 800, 600
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        return x, y
    # cv2.moveWindow(window_name, *get_screen_size())

    # face_detector = FaceDetector(model_name="haarcascades")
    # face_detector = FaceDetector(model_name="dnn")
    face_detector = FaceDetector(model_name="yolov11")
    face_tracker = FaceTracker()

    video_paths = discover_video_files(INPUT_PATH)

    video_iter = tqdm(video_paths, desc="Processing videos")
    for video_path in video_iter:
        movie_name = os.path.basename(video_path)
        movie_stem = os.path.splitext(movie_name)[0]

        # Mirror the input directory structure under OUTPUT_PATH
        relative_dir = os.path.relpath(os.path.dirname(video_path), INPUT_PATH)
        out_path = os.path.join(OUTPUT_PATH, relative_dir)
        os.makedirs(out_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        # Skip if already processed
        # Divide by 2 just to make sure at least half of the video is processed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / 2
        if os.path.exists(os.path.join(out_path, movie_stem, "frames")):
            # update_face_trackers now only runs once every TRACKER_STEP real
            # frames, so saved outputs land every SAVE_DATA_INTERVAL calls,
            # i.e. every SAVE_DATA_INTERVAL * TRACKER_STEP real frames.
            if int(total_frames / (SAVE_DATA_INTERVAL * TRACKER_STEP)) < len(os.listdir(os.path.join(out_path, movie_stem, "frames"))):
                cap.release()
                continue

        os.makedirs(os.path.join(out_path, movie_stem, "frames"), exist_ok=True)

        face_tracker.reset()

        if hasattr(face_tracker, "whisper_wrapper"):
            # Extract audio features
            # Load video
            try:
                # clip = VideoFileClip(video_path)
                clip = safe_videoclip(video_path)
            except:
                cap.release()
                continue
            # Extract audio as numpy array
            try:
                audio = clip.audio.to_soundarray(fps=16000)  # force 16 kHz
            except:
                cap.release()
                continue
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Convert to torch tensor
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
            # emotion_audio = face_tracker.fer.inference_audio(waveform)
            audio_encoder_outputs = face_tracker.whisper_wrapper.extract_audio_features(waveform)
            torch.save(
                audio_encoder_outputs,
                os.path.join(out_path, movie_stem, "audio_features.pt")
            )

            clip.close()
            # print(emotion_audio)

        # Extract video features in bounded-size windows instead of decoding
        # the whole video into RAM up front. Long videos can otherwise hold
        # many GB of raw frames in memory at once; the tracker's own state
        # (face_tracker.tracker / .physio / .frame_count) already persists
        # across windows, so identities and rPPG history carry over normally.
        # Only a handful of frames right at each window boundary (< step
        # frames, see compute_trackers) go untracked, the same edge effect
        # that previously only happened once at the very end of the video.
        # Temporary per-stage timing so we can see exactly where time is
        # going (added while chasing a reported slowdown) -- remove once
        # that's diagnosed.
        stage_times = defaultdict(float)

        batch_size = 64
        while True:
            t0 = time.time()
            frames = []
            for _ in range(FRAME_WINDOW_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
            stage_times["read"] += time.time() - t0

            if not frames:
                break

            t0 = time.time()
            faces = []
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]

                faces.extend(face_tracker.compute_face(frame_batch=batch))
            stage_times["detect"] += time.time() - t0

            # Compute face trackers for this window
            t0 = time.time()
            output_bbs, output_face_ids = face_tracker.compute_trackers(faces, frames)
            stage_times["track"] += time.time() - t0

            # Update all trackers for this window. Only every TRACKER_STEP-th
            # entry is a genuine DeepSort detection (compute_trackers fills the
            # frames in between with interpolated boxes) -- these are also the
            # indices this stride lands on, so update_face_trackers now always
            # runs on a real detection instead of an interpolated one, and
            # skips the BGR->RGB conversion + physio queueing for the other
            # 4 out of every 5 frames.
            t0 = time.time()
            for i in range(0, len(output_bbs), TRACKER_STEP):
                face_tracker.update_face_trackers(frames[i], output_bbs[i], output_face_ids[i], movie_stem, out_path)

                face_tracker.frame_count += 1

                # resized_frame = cv2.resize(frames[i], (frames[i].shape[1]//2, frames[i].shape[0]//2))
                # cv2.imshow(window_name, resized_frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            stage_times["update"] += time.time() - t0

            if len(frames) < FRAME_WINDOW_SIZE:
                break

        cap.release()

        # Make sure every background JPEG write for this video has finished
        # (and surface any write errors) before moving on to the next one.
        t0 = time.time()
        face_tracker.wait_for_pending_saves()
        stage_times["io_wait"] += time.time() - t0

        face_tracker.export_data(out_path, movie_stem)

        video_iter.set_postfix({k: f"{v:.1f}s" for k, v in stage_times.items()})

    face_tracker.shutdown()
