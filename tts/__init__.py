import torch

def load_tacotron2_model():
    return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')

def load_waveglow_model():
    return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')

def load_utils():
    return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')