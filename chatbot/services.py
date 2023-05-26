import torch
from .preprocessing.text import Tokenizer
import re

class ChatbotService:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer("./chatbot/tokenizer/tokenizer.pkl")
        self.end_token = self.tokenizer.dictionary.index('<end>')
        self.model = torch.jit.load("./chatbot/saved_models/model.pt")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x: torch.Tensor, num_tokens: int, end_token: int) -> torch.Tensor:
        for _ in range(num_tokens):
            output = self.model(x)

            preds = output[:, -1, :]
            _, predict_token = torch.max(preds, dim=-1)
            
            if predict_token == end_token:
                break
            x = torch.concat([x, predict_token.unsqueeze(0)], dim=-1)
        return x
    
    def response(self, message: str):
        digits = self.tokenizer.text_to_sequences([message], start_token=True, sep_token=True)
        len_seq = digits.shape[1]

        digits = torch.tensor(digits).to(self.device)
        result = self.predict(digits, 256, self.end_token)
        result = result.cpu()
        torch.cuda.empty_cache()
        seq = []
        for word in result[0, len_seq:]:
            seq.append(self.tokenizer.dictionary[word.item()])
        seq = "".join(seq)
        seq = re.sub("</w>", " ", seq)
        seq = re.sub("<list>", "&#x2022; ", seq)
        seq = re.sub(" i ", " I ", seq)
        seq = re.sub(" ai ", "A I", seq)
        seq = self.tokenizer.decode_special_tokens(seq)
        seq = seq.capitalize()
        return seq