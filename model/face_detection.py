from abc import ABC, abstractmethod

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


class AbstractModel(ABC):
    @abstractmethod
    def inference():
        return
        
class HaarCascades(AbstractModel):
    def __init__(self):
        self.face_detector_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
    def inference(self, input_frame):
        gray = cv2.cvtColor(input_frame.copy(), cv2.COLOR_BGR2GRAY)
        
        return self.face_detector_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
class DNN(AbstractModel):
    CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self):
        config_path = "model_checkpoints/deploy.prototxt"
        model_path = "model_checkpoints/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        
        self.face_detector_model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        
    def inference(self, input_frame):
        h, w = input_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(input_frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_detector_model.setInput(blob)
        detections = self.face_detector_model.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
                
        return boxes

  
class MyYOLO(AbstractModel):
    def __init__(self, device="cuda"):
        self.model = YOLO("model_checkpoints/yolov11s-face.pt").to(device=device)
        self.conf = 0.5 # default = 0.7
        
    def inference(self, input_frame):
        results = self.model.predict(input_frame, conf=self.conf, verbose=False)
        
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            
        return boxes.xyxy.to(device="cpu").numpy().tolist()
    
    def inference_batch(self, input_frame_batch):
        results = self.model.predict(input_frame_batch, conf=self.conf, verbose=False)
        
        # Process results list
        boxes_list = []
        for result in results:
            boxes = result.boxes.xyxy.to(device="cpu").numpy().tolist() # Boxes object for bounding box outputs
            boxes_list.append(boxes)
            
        return boxes_list

class FaceDetector(object):
    def __init__(self, model_name):
        if model_name == "haarcascades":
            self.model = HaarCascades()
        elif model_name == "dnn":
            self.model = DNN()
        elif model_name == "yolov11":
            self.model = MyYOLO()
        else:
            raise NotImplementedError(f"Model {model_name} not implemented!")
        
        self.face_threshold = 0.005 # default: 0.05
        
    def inference(self, **kwargs):
        h, w, _ = kwargs["input_frame"].shape
        pred_boxes = self.model.inference(**kwargs)
        
        faces = []
        for (x1, y1, x2, y2) in pred_boxes:
            if x2 - x1 > self.face_threshold * w and y2 - y1 > self.face_threshold * h:
                faces.append([round(x1), round(y1), round(x2-x1), round(y2-y1)])
            
        return faces
    
    def inference_batch(self, **kwargs):
        h, w, _ = kwargs["input_frame_batch"][0].shape
        pred_boxes_list = self.model.inference_batch(**kwargs)
        
        faces_list = []
        for pred_boxes in pred_boxes_list:
            if len(pred_boxes) == 0:
                faces_list.append([])
            else:
                faces = []
                for (x1, y1, x2, y2) in pred_boxes:
                    if x2 - x1 > self.face_threshold * w and y2 - y1 > self.face_threshold * h:
                        faces.append([round(x1), round(y1), round(x2-x1), round(y2-y1)])
                        
                faces_list.append(faces)
                
        return faces_list
