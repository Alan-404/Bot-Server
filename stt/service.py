import torch
import numpy as np
from .preprocessing import Decoder, prepare_model_input
class STTService:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load("./stt/saved_models/whisper.jit")
        self.model.to(self.device)

        self.decoder = Decoder("./stt/tokenizer/dictionary.pkl")

    def transcribe(self, samples: torch.Tensor):
        data_input = prepare_model_input(samples, device = self.device)
        output = self.model(data_input)

        result = []
        for item in output:
            result.append(self.decoder(item.cpu()))
        return " ".join(result)
    def transform_samples(self, samples: np.ndarray):
        return self.transcribe(torch.tensor(samples).unsqueeze(0))
    