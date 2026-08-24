import time
from PIL import Image

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

import hopenet.hopenet as hn
from hopenet.utils import draw_axis


class HeadPose:
    def __init__(self):
        self.transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ResNet50 structure
        self.model = hn.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Load snapshot
        saved_state_dict = torch.load(
            "model_checkpoints/hopenet_robust_alpha1.pkl",
            map_location=torch.device("cuda")
        )
        self.model.load_state_dict(saved_state_dict, strict=True)
        self.model.to(device="cuda")
        self.model.eval()
        
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda()
        
    def get_head_pose(self, frame, bbox, time_logging=None):
        x, y, w, h = bbox
        
        # TODO: extend proportional
        x_min = x - 100
        x_max = x + w + 100
        y_min = y - 100
        y_max = y + h + 100
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        crop = frame[y_min:y_max,x_min:x_max]

        # Transform
        start_time = time.time()
        img = self.transformations(crop)
        if time_logging is not None:
            time_logging["head_pose_transform"].append(time.time()-start_time)
        
        start_time = time.time()
        with torch.no_grad():
            yaw, pitch, roll = self.model(img.unsqueeze(0).to(device="cuda"))
        if time_logging is not None:
            time_logging["head_pose_inference"].append(time.time()-start_time)

        start_time = time.time()
        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)
        
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
        if time_logging is not None:
            time_logging["head_pose_postprocess"].append(time.time()-start_time)
        
        start_time = time.time()
        yaw_predicted = yaw_predicted.to(device="cpu", non_blocking=True).numpy()
        pitch_predicted = pitch_predicted.to(device="cpu", non_blocking=True).numpy()
        roll_predicted = roll_predicted.to(device="cpu", non_blocking=True).numpy()
        if time_logging is not None:
            time_logging["head_pose_tocpu"].append(time.time()-start_time)
        
        return yaw_predicted.astype(np.float16), pitch_predicted.astype(np.float16), roll_predicted.astype(np.float16)
    
    def draw_headpose(self, frame, hp, bbox):
        x, y, w, h = bbox
        
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h
        
        yaw_predicted, pitch_predicted, roll_predicted = hp
                        
        draw_axis(
            frame,
            yaw_predicted,
            pitch_predicted,
            roll_predicted,
            tdx = (x_min + x_max) / 2,
            tdy= (y_min + y_max) / 2,
            size = h
        )
