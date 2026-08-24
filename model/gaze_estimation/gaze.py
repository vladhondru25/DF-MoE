import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.gaze_estimation.helpers import get_model, draw_bbox_gaze


configs = {
    "resnet34": {
        "bins": 90,
        "binwidth": 4,
        "angle": 180  # angle range
    }
}


class GazeEstimator():
    def __init__(self, model, device="cuda"):
        dataset_config = configs[model]
        self.bins = dataset_config["bins"]
        self.binwidth = dataset_config["binwidth"]
        self.angle = dataset_config["angle"]
        
        self.device = device
        
        self.gaze_detector = get_model(model, self.bins, inference_mode=True)
        state_dict = torch.load(os.path.join("model_checkpoints", f"{model}-gaze.pt"), map_location=self.device)
        self.gaze_detector.load_state_dict(state_dict)
        
        self.gaze_detector.to(self.device)
        self.gaze_detector.eval()

        self.idx_tensor = torch.arange(self.bins, device=self.device, dtype=torch.float32)
        
    def pre_process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        image_batch = image.unsqueeze(0)
        return image_batch
    
    def inference(self, frame, bbox):
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
        
        # Inference on gaze
        crop = self.pre_process(crop)
        crop = crop.to(device=self.device)
        with torch.no_grad():
            pitch, yaw = self.gaze_detector(crop)

        pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

        # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle

        # Degrees to Radians
        pitch_predicted = np.radians(pitch_predicted.to(device="cpu").item())
        yaw_predicted = np.radians(yaw_predicted.to(device="cpu").item())

        return pitch_predicted, yaw_predicted

    def draw_gaze(self, frame, bbox, pitch_predicted, yaw_predicted) -> None:
        x, y, w, h = bbox
        new_bbox = [x, y, x+w, y+h]
        
        # draw box and gaze direction
        draw_bbox_gaze(frame, new_bbox, pitch_predicted, yaw_predicted)
