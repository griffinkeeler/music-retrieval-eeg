"""Frozen MERT feature extraction for aligned music windows."""

import librosa
import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MERTFeatureExtractor(nn.Module):
    """Encode audio waveforms with the final hidden layer of MERT."""

    def __init__(self, model_name="m-a-p/MERT-v1-95M"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.requires_grad_(False)
        self.target_sr = self.processor.sampling_rate

    def encode_arrays(self, arrays):
        """Encode in-memory waveforms sampled at ``target_sr``."""
        inputs = self.processor(
            arrays,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]  # [batch, time, 768]
        return hidden.transpose(1, 2).contiguous()  # [batch, 768, time]

    def forward(self, audio_paths: list[str]):
        arrays = [librosa.load(path, sr=self.target_sr)[0] for path in audio_paths]
        return self.encode_arrays(arrays)
