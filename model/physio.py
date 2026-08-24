from collections import deque

import torch
import torchvision.transforms as transforms

import clip
from physio_measurements.full_network import VL_phys


class Physio(object):
    def __init__(self, device="cuda"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711))
        ])
        tokenizer = clip.tokenize
        
        self.device = torch.device(device=device)
        
        self.model = VL_phys(device=device, model_path="model_checkpoints/physio_model.bin").to(device)
        self.model.eval()
        
        nums = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
        text_template = (
            "the frequency of the horizontal color variation on the left side is {} "
            "times of that on the right side of the image"
        )
        texts = [text_template.format(num) for num in nums]
        tokenized_texts = [tokenizer(text) for text in texts]
        self.tokenized_texts = torch.cat(tokenized_texts, dim=0)
        
        self.image_queue = {}
        
    def reset_image_queue(self):
        self.image_queue = {}
        
    def add_image_to_queue(self, frame, bbox, face_id):
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
        
        if face_id not in self.image_queue:
            self.image_queue[face_id] = deque(maxlen=11)
            
        self.image_queue[face_id].append(self.transform(crop))

    def inference(self, face_id):
        if len(self.image_queue[face_id]) < self.image_queue[face_id].maxlen:
            return
        
        images = torch.stack([image for image in self.image_queue[face_id]]).unsqueeze(0)

        # Inference
        with torch.no_grad():
            images = images.to(self.device)
            texts = self.tokenized_texts.unsqueeze(0).to(self.device)
            
            # signals, img_features_text, text_features, rec_patches, mask, gt_patches = self.model(images, texts)
            signals, _, _, _, _, _ = self.model(images, texts)
            
            signals = signals.squeeze(0).to(device="cpu", non_blocking=True).numpy()
            
            return signals
        
