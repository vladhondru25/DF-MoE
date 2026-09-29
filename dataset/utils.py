import numpy as np
import cv2

def align_audio_to_frames_by_fps(
    audio_feats,         
    frame_indices,          
    fps,                 
    hop_ms=20.0,         
    t0_video=0.0,        
    t0_audio=0.0,        
    method="nearest",    
    pool=None,           
    pool_radius_ms=10.0  
):
    # print(audio_feats.shape)
    T_a, D = audio_feats.shape
    audio_rate = 1000.0 / hop_ms
    frame_times = t0_video + np.array(frame_indices) / float(fps)

    audio_pos = (frame_times - t0_audio) * audio_rate

    if pool is None:
        if method == "nearest":
            idx = np.rint(audio_pos).astype(int)
        elif method == "floor":
            idx = np.floor(audio_pos).astype(int)
        elif method == "ceil":
            idx = np.ceil(audio_pos).astype(int)

        idx = np.clip(idx, 0, T_a - 1)
        aligned = audio_feats[idx]                         

    else:
        r_steps = int(np.round((pool_radius_ms / 1000.0) * audio_rate))
        idx_center = np.rint(audio_pos).astype(int)
        idx_center = np.clip(idx_center, 0, T_a - 1)

        out = np.empty((num_frames, D), dtype=audio_feats.dtype)
        for i, c in enumerate(idx_center):
            lo = max(0, c - r_steps)
            hi = min(T_a, c + r_steps + 1)
            if pool == "mean":
                out[i] = audio_feats[lo:hi].mean(axis=0)
            elif pool == "max":
                out[i] = audio_feats[lo:hi].max(axis=0)
        aligned = out

    return aligned

def get_fps_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(fps)