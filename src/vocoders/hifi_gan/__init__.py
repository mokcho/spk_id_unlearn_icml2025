import json
import torch
import torch.nn as nn
from utils import AttrDict
from data.melspec import MAX_WAV_VALUE
from .models import Generator


# Import HIFI-GAN Vocoder
class HIFIGAN(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super().__init__()
        # Load Config File
        with open(config_path, "r") as f:
            config = json.load(f)
        config = AttrDict(config)

        # Load checkpoints
        state_dict = torch.load(ckpt_path, map_location='cpu')['generator']

        # Define Generator
        self.generator = Generator(config)
        self.generator.load_state_dict(state_dict)
        self.generator.eval()
        self.generator.remove_weight_norm()
    
    def forward(self, mel, to_int=False):
        audio = self.generator(mel)
        if to_int:
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
        return audio