import os
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# INPUT_PATH = "/home/vhondru/vhondru/phd/biodeep/BiodeepDetection/interim_outputs/english"
INPUT_PATH = "/media/vhondru/hdd/biodeep/interim_outputs/english"
BATCH_SIZE = 32


class VisualFeatureExtractor(object):
    def __init__(self, backbone="dinov3"):
        if backbone == "dinov3":
            pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"

            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.model = AutoModel.from_pretrained(
                pretrained_model_name, 
                device_map="auto", 
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
    def extract_features(self, images) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # pooled_output = outputs.pooler_output
        # print("Pooled output shape:", pooled_output.shape)
        # print("last_hidden_state:", outputs.last_hidden_state.shape)

        return outputs.last_hidden_state.cpu()
    

def batch_scroll(lst):
    for i in range(0, len(lst), BATCH_SIZE):
        yield lst[i:i + BATCH_SIZE]

def main() -> None:
    visual_features_extractor = VisualFeatureExtractor("dinov3")

    for method in os.listdir(INPUT_PATH):
        for video in tqdm(os.listdir(os.path.join(INPUT_PATH, method)), desc=f"Processing {method}"):
            out_dir = os.path.join(INPUT_PATH, method, video, "visual_features")
            os.makedirs(out_dir, exist_ok=True)

            frames_list = os.listdir(os.path.join(INPUT_PATH, method, video, "frames"))
            for frames_batch in batch_scroll(frames_list):
                frames_paths = [os.path.join(os.path.join(INPUT_PATH, method, video, "frames"), frame_name) for frame_name in frames_batch]
                frames_out_paths = [os.path.join(out_dir, frame_name[:-3] + "pt") for frame_name in frames_batch]

                if all([os.path.exists(out_path) for out_path in frames_out_paths]):
                    continue
                
                frames_pil = [Image.open(frame_path) for frame_path in frames_paths]

                last_hidden_states = visual_features_extractor.extract_features(frames_pil)

                for out_path, last_hidden_state in zip(frames_out_paths, last_hidden_states):
                    torch.save(last_hidden_state, out_path)

        #         break

        #     break
        # break


if __name__ == "__main__":
    main()
