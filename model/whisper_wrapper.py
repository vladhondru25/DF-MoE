import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel


class WhisperWrapper(object):
    def __init__(self):
        # Load processor + model (not the ForConditionalGeneration, just the encoder/decoder model)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperModel.from_pretrained("openai/whisper-tiny")
        
    def extract_audio_features(self, waveform):
        # Load audio (16kHz mono is expected)
        # waveform, sr = torchaudio.load("your_audio.wav")
        # if sr != 16000:
        #     waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Convert to log-Mel spectrogram features
        # inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

        # print("Input features shape:", inputs.input_features.shape) 
        # (batch_size, feature_dim=80, sequence_length) torch.Size([1, 80, 3000])

        # Pass through encoder to get hidden states
        with torch.no_grad():
            encoder_outputs = self.model.encoder(inputs.input_features)

        # print("Encoder hidden states shape:", encoder_outputs.last_hidden_state.shape)  
        # (batch_size, seq_len, hidden_dim=384 for tiny) torch.Size([1, 1500, 384])
        
        return encoder_outputs.last_hidden_state.to(device="cpu")
