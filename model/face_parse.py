#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from face_parsing.model import BiSeNet


class FaceParser(object):
    def __init__(self):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        self.net.load_state_dict(torch.load("model_checkpoints/faceparser.pth"))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Colors for all 20 parts
        self.part_colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0],
            [255, 0, 85], [255, 0, 170],
            [0, 255, 0], [85, 255, 0], [170, 255, 0],
            [0, 255, 85], [0, 255, 170],
            [0, 0, 255], [85, 0, 255], [170, 0, 255],
            [0, 85, 255], [0, 170, 255],
            [255, 255, 0], [255, 255, 85], [255, 255, 170],
            [255, 0, 255], [255, 85, 255], [255, 170, 255],
            [0, 255, 255], [85, 255, 255], [170, 255, 255]
        ]
        
        self.parts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        
        # Indexing starts from 1
        # 1  face
        # 10 nose
        # 11 teeth
        # 12 upper lip
        # 13 lower lip
        # 17 hair

    def vis_parsing_maps(self, im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
        im = np.array(im)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        if save_im:
            cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
            cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return vis_im

    def inference(self, frame, bbox):
        x, y, w, h = bbox
        
        # TODO: extend proportional
        x_min = x - 30
        x_max = x + w + 30
        y_min = y - 30
        y_max = y + h + 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        crop = frame[y_min:y_max,x_min:x_max]
        
        img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            # img = Image.open(osp.join(dspth, image_path))
            # img_pil = img_pil.resize((512, 512), Image.BILINEAR)
            img_tensor = self.to_tensor(img_pil)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            img_tensor = img_tensor.to(device="cuda")
            
            out = self.net(img_tensor)[0]
            
            parsing = out.squeeze(0).to(device="cpu").numpy().argmax(0)

            return parsing, img_pil
