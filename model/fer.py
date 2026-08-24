import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from speechbrain.inference.interfaces import foreign_class
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


class FER(object):
    def __init__(self, model_name=None, device="cuda", use_audio=False):
        if model_name is None:
            model_name = get_model_list()[0] # enet_b0_8_best_vgaf

        self.fer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)
        
        self.use_audio = use_audio
        if use_audio:
            # Method 1
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
            self.audio_classifier = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

            self.audio_classifier.lm_head = torch.nn.Linear(in_features=1024, out_features=7)
            old_weights = torch.load("wav2vec-english-speech-emotion-recognition/pytorch_model.bin")
            lm_head_weights = {
                "weight": old_weights["classifier.out_proj.weight"],
                "bias": old_weights["classifier.out_proj.bias"],
            }
            self.audio_classifier.lm_head.load_state_dict(lm_head_weights)

            # Method 2
            # self.audio_classifier = foreign_class(
            #     source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            #     hparams_file="hyperparams.yaml",
            #     pymodule_file="custom_interface.py",
            #     classname="CustomEncoderWav2vec2Classifier",
            #     run_opts={"device": "cuda"}
            # )

    def inference_video(self, frame, bbox):
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
        
        emotion_video = None
        emotion_audio = None
        
        # Inference
        emotion_video, _ = self.fer.predict_emotions(crop, logits=True)
        
        # print(emotion)
        return emotion_video
    
    def inference_audio(self, waveform):
        if not self.use_audio:
            raise ValueError("use audio not implemented for emotion recognition")
        
        # Method 1
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.audio_classifier(inputs.input_values.squeeze(0)).logits
            predictions = torch.nn.functional.softmax(outputs.mean(dim=1), dim=-1)  # Average over sequence length
            predicted_label = torch.argmax(predictions, dim=-1)
            emotion_audio = self.audio_classifier.config.id2label[predicted_label.item()]
            
        # Method 2
        # out_prob, score, index, text_lab = self.audio_classifier.classify_batch(waveform)
        # emotion_audio = text_lab
        
        return emotion_audio
