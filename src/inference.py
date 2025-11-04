import heapq
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from glob import glob
from jiwer import wer
from multiprocessing import get_context
from omegaconf import OmegaConf
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, HubertForCTC, Wav2Vec2ProcessorWithLM
from tqdm import tqdm

from data.melspec import mel_spectrogram, load_wav
from data.utils import get_phoneme_frome_textgrid, get_duration
from eval.wer import *
from pipelines.duration_predictor import DurationPredictorPipline
from pipelines.zero_shot_tts import ZeroShotTTSPipline
from speaker_verification.verification import _verification, init_model
from text import phoneme_to_sequence
from vocoders import get_vocoder

random.seed(1234) # Fix random seed
torch.manual_seed(1234) # Fix random seed

def get_audio_textgrid_path(filename):
    # the code is a little messy here,
    # and it assumes certain structure of file paths.
    # modify it as needed, you just want the specific output audio and textgrid paths.
    if ".wav" in filename :
        textgrid_path = filename.replace(".wav", ".TextGrid")
        textgrid_path = textgrid_path.replace(audio_root_path, textgrid_root_path)
        return filename, textgrid_path
    elif audio_root_path in filename: 
        return filename+".wav", filename.replace(audio_root_path, textgrid_root_path)+".TextGrid"
    else : 
        return os.path.join(audio_root_path, filename + ".wav"), os.path.join(textgrid_root_path, filename + ".TextGrid")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

device = "cuda"

################# Settings #################

# LibriHeavy Forget
eval_filelist_fp = "./filelists/libriheavy_forget_eval_pair.json"
audio_root_path =  "/data/libriheavy/train/large/" # path to audio
textgrid_root_path = "/data/libriheavy/TextGrid/train/large/" # path to textgrid

acoustic_ckpt_fp = "TGU_final.pt"

dp_ckpt_fp = "/data/unlearn_ckpt/duration_predictor.pt"

use_lm = False
use_duration_predictor = True # you can choose to use duration predictor or not

vocoder_ckpt = "/data/unlearn_ckpt/hifi_gan_final.pt"

############################################

# Output Paths

result_path = "TGU" #################
result_path = os.path.join("outputs/forget", result_path)
result_path_wav = os.path.join(result_path, "wavs")
result_path_spk_emb = os.path.join(result_path, "spk_embs")
make_dir(result_path)
make_dir(result_path_wav)
make_dir(result_path_spk_emb)

n = 3
n_select = 3
prompt_seq_len = 100 * 3 # 3 secs
alpha = 0.7 # CFG Factor
fp16 = True

with open(eval_filelist_fp, "r") as f:
    eval_filelist = json.load(f)

# Import Models
audio_cfg = "./configs/audio.yaml"
audio_cfg = OmegaConf.load(audio_cfg)
to_mel = partial(
            mel_spectrogram,
            n_fft=audio_cfg.fft_size,
            num_mels=audio_cfg.num_mels,
            sampling_rate=audio_cfg.sampling_rate,
            hop_size=audio_cfg.hop_size,
            win_size=audio_cfg.win_size,
            fmin=audio_cfg.fmin,
            fmax=audio_cfg.fmax,
            center=False
        )

# Duration Predictor
if use_duration_predictor :
    dp_ckpt = torch.load(dp_ckpt_fp, map_location='cpu')
    duration_predictor = DurationPredictorPipline(dp_ckpt['cfg']).to(device)
    duration_predictor.load_state_dict(dp_ckpt['model'])
    duration_predictor.eval()

# Acoustic Model
acoustic_ckpt = torch.load(acoustic_ckpt_fp, map_location='cpu')
acoustic_model = ZeroShotTTSPipline(acoustic_ckpt['cfg'],
                                    unlearn_layer=unlearn_layer,
                                    dtype=torch.float16 if fp16 else torch.float32,
                                    ).to(device)
acoustic_model.load_state_dict(acoustic_ckpt['model'], strict=False)
acoustic_model.eval()

# Vocoder
vocoder_cfg = "./configs/hifigan_config.json"
vocoder = get_vocoder("hifi-gan", vocoder_ckpt, vocoder_cfg).to(device)

# if fp16: # needs debugging
#     acoustic_model = acoustic_model.half()
#     duration_predictor = duration_predictor.half()
#     vocoder = vocoder.half()

#%%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
out_results = []
wers, sims = [], []
tq = tqdm(eval_filelist, desc=result_path)
for idx, filenames in enumerate(tq) :
    try : 
        src_data, tgt_data = filenames

        src_audio_fp, src_textgrid_fp = get_audio_textgrid_path(src_data)    
        tgt_audio_fp, tgt_textgrid_fp = get_audio_textgrid_path(tgt_data)
        
        name = src_data.split("/")[-1]
        name = name.replace("/", "_")
        
        if os.path.exists(os.path.join(result_path_spk_emb, name+"_src" + ".npy")) :
            continue
        
        try : 
            tgt_mel, tgt_phn_expanded, tgt_phn_ids, tgt_phn_durations = process_prompt(to_mel, tgt_audio_fp, tgt_textgrid_fp, device)
            tgt_segment, tgt_transcript = get_phoneme_frome_textgrid(tgt_textgrid_fp, return_transcript=True)
            src_mel, src_phn_expanded, src_phn_ids, src_phn_durations = process_prompt(to_mel, src_audio_fp, src_textgrid_fp, device)
            ref_segment, ref_transcript = get_phoneme_frome_textgrid(tgt_textgrid_fp, return_transcript=True)
        except : 
            print(f"skipping {src_data} with {tgt_data}")
            continue

        # Predicting Durations
        
        contain_spn = False
        for seg in tgt_segment :
            if 'spn' in seg[0]:
                contain_spn = True
                break
        if contain_spn :
            continue
        
        if use_duration_predictor :
            with torch.no_grad():
                tgt_phn_duration = duration_predictor.predict_dur(
                    src_durs=src_phn_durations,
                    src_phn_ids=src_phn_ids,
                    tgt_phn_ids=tgt_phn_ids,
                )
                tgt_phn_duration = torch.clamp_(tgt_phn_duration, 0)
            tgt_phn_expanded = torch.repeat_interleave(
                tgt_phn_ids,
                tgt_phn_duration.squeeze(0),
                dim=-1
            )
        tgt_phn_expanded = tgt_phn_expanded.unsqueeze(0)

        # Generate Mel-Spectrogram & Waveform from Acoustic Model + Vocoder
        src_mel = src_mel.repeat(n, 1, 1)
        src_phn_expanded = src_phn_expanded.repeat(n, 1)
        tgt_phn_expanded = tgt_phn_expanded.repeat(n, 1)

        with torch.no_grad():
            gen_mel = acoustic_model.generate_mel_batch(
                                        tgt_phn_expanded,
                                        src_mel,
                                        src_phn_expanded,
                                        alpha=alpha,
                                    ).float()
            gen_wav = vocoder(gen_mel).squeeze(1)

        # use vocoder to generate GT waveform from source mel
        with torch.no_grad():
            gt_wav = vocoder(src_mel).squeeze(1) # (n, TD length)

            # save speaker embeddings for SIM evaluation
            gen_emb = sv_model(gen_wav.float())
            gt_emb = sv_model(gt_wav.float())

        for i in indices :
            _name = name + f"_{i}"
            _emb_path = os.path.join(result_path_spk_emb, _name + ".npy")
            _wav_path = os.path.join(result_path_wav, _name + ".wav")
            
            _emb_dir_name = os.path.dirname(_emb_path)
            if not os.path.exists(_emb_dir_name):
                os.makedirs(_emb_dir_name, exist_ok=True)
            _wav_dir_name = os.path.dirname(_wav_path)
            if not os.path.exists(_wav_dir_name):
                os.makedirs(_wav_dir_name, exist_ok=True)

            sf.write(_wav_path, gen_wav[i].float().detach().cpu().numpy(), 16000)
            np.save(_emb_path, gen_emb[i].float().detach().cpu().numpy())

        _name = name + f"_src"
        _emb_path = os.path.join(result_path_spk_emb, _name + ".npy")
        _wav_path = os.path.join(result_path_wav, _name + ".wav")
        _emb_dir_name = os.path.dirname(_emb_path)
        if not os.path.exists(_emb_dir_name):
            os.makedirs(_emb_dir_name, exist_ok=True)
        _wav_dir_name = os.path.dirname(_wav_path)
        if not os.path.exists(_wav_dir_name):
            os.makedirs(_wav_dir_name, exist_ok=True)
        sf.write(_wav_path, gt_wav[0].float().detach().cpu().numpy(), 16000)
        np.save(_emb_path, gt_emb[0].float().detach().cpu().numpy())
    except :
        continue

print(f"FINAL")
print(f"Results are saved to {result_path}")
