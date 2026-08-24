import argparse
import os
import sys
import tempfile
import time
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import torch
from moviepy import ImageSequenceClip, VideoFileClip
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.face_detection import FaceDetector
from model.face_parse import FaceParser
from model.fer import FER
from model.gaze_estimation.gaze import GazeEstimator
from model.head_pose import HeadPose
from feature_extraction_scripts.extract_audio_emotions import load_audio_emotion_model, extract_centered_whisper_segments, DEF_LABELS
from model.physio import Physio
from model.whisper_wrapper import WhisperWrapper
from deep_sort_realtime.deepsort_tracker import DeepSort
from dataset.dataset_from_single_video import MavosFeatureSingleVideo
from test_scripts.test_moe import dict_collate_fn
from model.model_factory import create_model
from einops import rearrange
import torch.nn.functional as F
import json
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T
SAVE_DATA_INTERVAL = 2

class EmbedderWrapper():
    def __init__(self):
        self.face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')
        self.transforms = T.Compose([T.Resize((160, 160)), T.ToTensor()])
    def predict(self, crops):
        crops = [self.transforms(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in crops]
        batch_tensor = torch.stack(crops).float().to("cuda")
        with torch.no_grad():
            face_embeds = self.face_embedder(batch_tensor).cpu().numpy()
        return face_embeds
class FaceTracker(object):
    def __init__(self):
        self.head_pose = HeadPose()
        
        self.gaze_estimator = GazeEstimator("resnet34")
        
        self.fer = FER(use_audio=False)
        
        self.face_parser = FaceParser()
        
        self.physio = Physio()
        
        self.whisper_wrapper = WhisperWrapper()
        embedder = EmbedderWrapper()
        self.tracker = DeepSort(max_age=30, n_init=2, max_iou_distance = 0.2)
        # self.tracker.embedder = embedder
        
        self.audio_emotion_model = load_audio_emotion_model()
        
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
        
    def compute_face(self, frame=None, frame_batch=None):
        boxes_list = face_detector.inference_batch(input_frame_batch=frame_batch)

        bbs_list = []
        for boxes in boxes_list:
            bbs = [([x,y,w,h],0.7,"face") for (x,y,w,h) in boxes]
            
            bbs_list.append(bbs)
        
        return bbs_list
        
    def compute_trackers(self, bbs, frames):
        output_bbs, output_face_ids = [], []
        
        step = 5
        for i in range(0, len(bbs), step):
            start = i
            end = i + step
            
            if end >= len(bbs):
                break
            
            # Start
            start_bbs, start_face_ids = [], []
            self.face_trackers.clear()
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
            self.face_trackers.clear()
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
                
        return output_bbs, output_face_ids
            
    def update_face_trackers(self, frame, output_bbs, output_face_ids):
        for face_id, box in zip(output_face_ids, output_bbs):
            box = [int(p) for p in box]
            x1, y1, x2, y2 = box
            
            frame_og = frame.copy()
            # Physiological measurements
            if hasattr(self, "physio"):
                self.physio.add_image_to_queue(cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB), [x1, y1, x2-x1, y2-y1], face_id)
            
            if not (self.frame_count > SAVE_DATA_INTERVAL and self.frame_count % SAVE_DATA_INTERVAL == 0):
                continue

            # Obtain head pose and draw it
            if hasattr(self, "head_pose"):
                hp = self.head_pose.get_head_pose(cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB), [x1, y1, x2-x1, y2-y1])
            
            # Obtain gaze and draw it
            if hasattr(self, "gaze_estimator"):
                pitch_predicted, yaw_predicted = self.gaze_estimator.inference(cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB), [x1, y1, x2-x1, y2-y1])
            
            # Facial emotion
            if hasattr(self, "fer"):
                emotion_video = self.fer.inference_video(cv2.cvtColor(frame_og, cv2.COLOR_BGR2RGB), [x1, y1, x2-x1, y2-y1])
                
            # Physio
            if hasattr(self, "physio"):
                rPPG = self.physio.inference(face_id)
            
            # Face
            if hasattr(self, "face_parser"):
                parsing, img_pil = self.face_parser.inference(frame_og, [x1, y1, x2 - x1, y2 - y1])
                self.data_to_save[f"{face_id}_{self.frame_count}"] = {}
                self.data_to_save[f"{face_id}_{self.frame_count}"]['bbox_face'] = np.array(box).astype(int)
                self.data_to_save[f"{face_id}_{self.frame_count}"]['frames'] = np.array(
                img_pil.resize((512, 512), Image.LANCZOS))

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
                
    def add_emotion_audio(self, audio_features):
        """ Add audio-based emotion predictions to the saved data. """
        for frame_name,_ in self.data_to_save.items():
            frame_index = int(frame_name.split("_")[1])
            
            last_hidden_state = extract_centered_whisper_segments(audio_features, frame_index)
            
            with torch.no_grad():
                emotion_logits = self.audio_emotion_model.forward_precomputed(last_hidden_state.unsqueeze(0))
                emotion_logits = emotion_logits.numpy().astype(np.float16)
                emotion_pred = DEF_LABELS[np.argmax(emotion_logits)]
            
            self.data_to_save[frame_name]["emotion_audio"] = emotion_pred
            
    def format_video_features(self, fps_video: int, video_path: str) -> dict:
        """ Format the video features into a more efficient structure """
        signals = self.data_to_save

        keys = list(signals.keys())
        keys.sort(key=lambda x: int(x.split("_")[-1]))
        
        sequences = {}
        for key in keys:
            segmentation_maps = np.repeat(np.expand_dims(signals[key]['face_parse'], axis=-1), 3, axis=-1)
            
            rPPG = np.array(signals[key]['rPPG']).squeeze().astype(np.float32)
            if len(rPPG.shape)<2:
                rPPG=np.zeros((5, 512)).astype(np.float32)
            identity, frame_idx = key.split("_")
            if identity in sequences:
                sequences[identity]['hp'].append(np.array(signals[key]['hp']).astype(np.float16).tolist())
                sequences[identity]['frames'].append(signals[key]['frames'])
                sequences[identity]['gaze'].append(np.array(signals[key]['gaze']).astype(np.float16).tolist())
                sequences[identity]['emotion_video'].append(signals[key]['emotion_video'])
                sequences[identity]['emotion_audio'].append(signals[key]['emotion_audio'])
                sequences[identity]['rPPG'].append(rPPG)
                sequences[identity]['face_parse'].append(segmentation_maps)
                sequences[identity]['frame_idx_in_original_video'].append(int(frame_idx))
                sequences[identity]['frame_idx_in_sequence'].append(len(sequences[identity]['frame_idx_in_sequence']))

            else:
                sequences[identity]= {
                    'frames':[signals[key]['frames']],
                    'hp':[signals[key]['hp']],
                    'gaze':[signals[key]['gaze']], 'emotion_video':[signals[key]['emotion_video']],
                    'emotion_audio':[signals[key]['emotion_audio']],
                    'rPPG':[rPPG],
                    'face_parse':[segmentation_maps],
                    'frame_idx_in_original_video': [int(frame_idx)],
                    'frame_idx_in_sequence': [0],
                    'video_path': video_path
                    
                }
                
        for identity in sequences:
            sequences[identity]['face_parse'] = self.frames_to_mp4_bytes(sequences[identity]['face_parse'])
            sequences[identity]['frames'] = self.frames_to_mp4_bytes(sequences[identity]['frames'])
            sequences[identity]['identity'] = int(identity)
            sequences[identity]['fps'] = fps_video
        return sequences

    def frames_to_mp4_bytes(self, frames, fps=25):
        clip = ImageSequenceClip(frames, fps=fps)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        # write_videofile expects a filename or file-like object with a .write() method
        clip.write_videofile(tmp_path, codec="libx264", audio=False, logger=None)
        clip.close()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        # print(tmp_path)
        os.remove(tmp_path)
        return video_bytes


def safe_load_videoclip(path, retries=3, delay=0.5, **kwargs) -> None:
    """ Safely load a VideoFileClip with retries on OSError. """
    for attempt in range(1, retries+1):
        try:
            return VideoFileClip(path, **kwargs)
        except OSError as e:
            print(f"[WARN] Attempt {attempt} failed for {os.path.basename(path)}: {e}")
            time.sleep(delay)
    raise OSError(f"Failed to load {path} after {retries} attempts")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference video for face analysis.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model weights for deepfake detection. The name of this file will give the type of signal and implicitly the model used for prediction.")
    args = parser.parse_args()
    return args

def extract_and_save_crops(frames, all_bboxes, identities, output_dir="extracted_crops"):
    """
    frames: List of file paths or numpy arrays
    all_bboxes: List of lists, where each sublist contains bboxes for that frame
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_idx, (frame_source, bboxes) in enumerate(tqdm(zip(frames, all_bboxes))):
        # Load the frame if it's a path, otherwise use it as an array
        image = cv2.imread(frame_source) if isinstance(frame_source, str) else frame_source
        
        if image is None:
            print(f"Skipping frame {frame_idx}: Image not found.")
            continue

        for box_idx, bbox in enumerate(bboxes):
            # Coordinates: [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = map(int, bbox)

            # Crop using NumPy slicing: [y_start:y_end, x_start:x_end]
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue
            identity = identities[frame_idx][box_idx]
            # Save to disk
            filename = f"frame_{frame_idx}_box_{box_idx}_{identity}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, crop)
            # print(frame_idx, bboxes)
            # exit()
def load_from_json(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        threshold = data['metadata']['decision_threshold']
        intervals_data = data["intervals"]
        calibrated_confidences = [intv["confidence_accuracy"] for intv in intervals_data]
        edges = [intv["lower_bound"] for intv in intervals_data]
        edges.append(intervals_data[-1]["upper_bound"])
        bin_edges = np.array(edges)
        
        return bin_edges, calibrated_confidences, threshold

def predict_confidence(new_predictions, intervals, confidences):
     
        new_preds = np.array(new_predictions)
        bin_indices = np.digitize(new_preds, intervals, right=True)
        bin_indices = np.clip(bin_indices, 1, len(intervals) - 1)
        
        mapped_confidences = [confidences[idx - 1] for idx in bin_indices]
        return np.array(mapped_confidences)
if __name__ == "__main__":
    face_detector = FaceDetector(model_name="yolov11")
    face_tracker = FaceTracker()

    args = parse_args()

    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    face_tracker.reset()
    
    # Extract audio features
    print("Extracting audio features...")
    
    # Load video
    try:
        clip = safe_load_videoclip(video_path)
    except:
        raise RuntimeError(f"Failed to load video.")
    
    # Extract audio as numpy array
    try:
        audio = clip.audio.to_soundarray(fps=16000)  # force 16 kHz
    except:
        raise RuntimeError(f"Failed to extract audio from video.")
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Convert to torch tensor
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
    audio_encoder_outputs = face_tracker.whisper_wrapper.extract_audio_features(waveform) # audio_features.pt

    clip.close()
                    
    # Extract video features
    print("Extracting video frames...")
    frames = []
    index = 0
    while True:
        ret, frame = cap.read()
        index+=1
        if not ret:
            break
        if index % 2 ==0:
            frames.append(frame)
    cap.release()

    faces = []
    batch_size = 64
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        
        faces.extend(face_tracker.compute_face(frame_batch=batch))
    
    # Compute face tracketes
    print("Computing face trackers...")
    output_bbs, output_face_ids = face_tracker.compute_trackers(faces, frames)
        
    # Extract video features
    # Update all trackers
    for i, (output_bb, output_face_id) in enumerate(tqdm(zip(output_bbs, output_face_ids), desc="Extracting video features")):
        try:
            face_tracker.update_face_trackers(frames[i], output_bb, output_face_id)
        except:
            continue 
        face_tracker.frame_count += 1
        if face_tracker.frame_count>1000:
            break
        
    face_tracker.add_emotion_audio(audio_encoder_outputs[0])
    
    # Inference model - MoE
    audio_features = audio_encoder_outputs.cpu().numpy().squeeze()
    feature_types_ds=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth", "full_frames", "raw_audio_features"]
    video_features = face_tracker.format_video_features(fps_video, video_path)
    output = {}
    
    model, feature_type_model, model_type = create_model(args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device="cuda" if torch.cuda.is_available() else "cpu"
    final_identities = {}
    for identity in list(video_features.keys()):
        if len(video_features[identity]['frame_idx_in_original_video']) >= 10:
            final_identities[identity] = video_features[identity]
    print(f"Found {len(final_identities)} identities")
    pbar = tqdm(list(final_identities.keys()), desc="Processing identities")
    result = {}
    identities = {}
    intervals, confidences, decision_threshold = load_from_json("config/config.json")
    for identity in pbar:
        ds = MavosFeatureSingleVideo(audio_features, video_features[identity],
                                40,
                                60,
                                feature_types_ds,
                                True, video_path)
        print(f"Identity {identity} has {len(ds)} sequences to process")
        test_dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8, collate_fn=dict_collate_fn)
        final_probabilities = []
        
        image = video_features[identity]['frames'][0]
        image = rearrange(image, 'c h w-> h w c')
        identities[identity] = Image.fromarray(image.numpy().astype(np.uint8))
        for batch in tqdm(test_dl):
            # print(list(batch.keys()), feature_types)
            
            batch['rPPG'] = torch.mean(batch['rPPG'], dim=2)
            if 'full_frames' in batch:
                batch['full_frames'] = rearrange(batch['full_frames'], 'b s c h w->b c s h w')
            if 'face_parse' in batch:
                face_parse = rearrange(batch['face_parse'], 'b s c h w->b s h w c')
                # for i, frame in enumerate(face_parse[0]):
                #     Image.fromarray(frame.numpy().astype(np.uint8)*12).save(f"face_parsing_test/image_{i}.png")
                
                batch['face_parse'] = torch.mean(batch['face_parse'].float(), dim=2).unsqueeze(dim=2)
                batch['face_parse'] = rearrange(batch['face_parse'], 'b s c h w->b c s h w')
                batch['face_parse'] = F.interpolate(batch['face_parse'], (batch['face_parse'].shape[2], 224, 224))
            if 'padding_mask' in batch:
                batch['padding_mask'] = batch['padding_mask'].to(device)
            input_dict = {tof: batch[tof].float().to(device) for tof in feature_type_model}
            with torch.no_grad():
                if len(feature_type_model) > 3:
                    input_dict['padding_mask'] = batch['padding_mask']
                    logits, _ = model(input_dict)
                    # logits = logits.squeeze(dim=-1)
                else:
                    input_model = [input_dict[key] for key in feature_type_model]
                    logits = model(*input_model).squeeze(dim=-1)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            final_probabilities.extend(probabilities)
        list_floats = list(1-np.array(final_probabilities))
        final_probabilities = [str(score) for score in list_floats]
        confidence = predict_confidence([1-np.mean(list_floats)], intervals, confidences)
        decision =  "Fake" if 1-np.mean(list_floats) < decision_threshold else "Real"
        result[identity] = {"fakeness score per sequence": final_probabilities, "identity fakeness score":str(np.mean(list_floats)), "decision": decision , "confidence": confidence[0]}


    output_dir = f"results/{os.path.basename(video_path)[:-4]}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/identities", exist_ok=True)
    for identity in identities:
        identities[identity].save(os.path.join(f"{output_dir}/identities/{identity}.png"))
    with open(f"{output_dir}/result_{model_type}.json", "w") as f:
        json.dump(result, f, indent=4)
    