from .diffwave_vocoder.inference import DiffWave
from .hifi_gan import HIFIGAN


def get_vocoder(name, ckpt_path, config_path=None, **kwargs):
    if name == 'hifi-gan':
        return HIFIGAN(config_path, ckpt_path)
    else:
        raise ValueError(f"Not Defined Vocoder {name}")