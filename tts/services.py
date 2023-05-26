import torch
import re
from . import load_tacotron2_model, load_waveglow_model, load_utils

class TTSService:
    def __init__(self) -> None:
        self.spect_model = load_tacotron2_model()
        self.spect_model = self.spect_model.to('cuda')
        self.spect_model.eval()

        self.wave_model = load_waveglow_model()
        self.wave_model = self.wave_model.remove_weightnorm(self.wave_model)
        self.wave_model = self.wave_model.to('cuda')
        self.wave_model.eval()

        self.utils = load_utils()
    def transform(self, text: str):
        text = re.sub(r"([:./,?!@#$%^&=`~*])", "", text)
        sequences, lengths = self.utils.prepare_input_sequence([text])
        with torch.no_grad():
            mel, _, _ = self.spect_model.infer(sequences, lengths)
            audio = self.wave_model.infer(mel)
        return audio[0].data.cpu().numpy()
        