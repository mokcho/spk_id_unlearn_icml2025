import heapq
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from jiwer import wer
import librosa
from multiprocessing import get_context
from pyctcdecode import build_ctcdecoder
from functools import partial
from omegaconf import OmegaConf


from text import phoneme_to_sequence
from data.utils import get_phoneme_frome_textgrid, get_duration
from data.melspec import mel_spectrogram, load_wav
from transformers import Wav2Vec2Processor, HubertForCTC, Wav2Vec2ProcessorWithLM


def remove_specific_special_characters(text):
    special_characters = '"\'?!.' 
    for char in special_characters:
        if char == "'" :
            text = text.replace(char,' ')
        else:
            text = text.replace(char, '') 
    return text

def get_audio_textgrid_path(filename, audio_root_path, textgrid_root_path):
    return os.path.join(audio_root_path, filename.split("/")[-1]  + "_src.wav"), os.path.join(textgrid_root_path, filename + ".TextGrid")


def sample_audio(audio_fp, tgt_sr=16000):
    audio, sr = load_wav(audio_fp, return_float32=True, do_normalize=True)
    if sr != tgt_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
    return audio

def process_prompt(to_mel, audio_fp, textgrid_fp, device):
    
    # Load Audio
    audio = sample_audio(audio_fp, tgt_sr=16000)
    audio = torch.from_numpy(audio).float()
    audio = audio.unsqueeze(0) # [1, seq_len]
    mel = to_mel(audio)

    # Load TextGrid
    segments = get_phoneme_frome_textgrid(textgrid_fp)
    phn_ids = [p[0] for p in segments]
    phn_ids = phoneme_to_sequence(phn_ids)

    phn_durations = get_duration(segments, mel.size(-1), 100)
    if phn_durations[-1] < 0 :
        phn_durations = phn_durations[..., :-1]
        phn_ids = phn_ids[:-1]
    phn_ids = torch.LongTensor(phn_ids).to(device)
    phn_durations = torch.LongTensor(phn_durations).to(device)
    phn_expanded = torch.repeat_interleave(
                                        phn_ids,
                                        phn_durations,
                                        dim=0)
    shorter_len = min(phn_expanded.size(-1), mel.size(-1))
    phn_expanded = phn_expanded[..., :shorter_len].to(device)
    mel = mel[..., :shorter_len].to(device)

    return mel, phn_expanded, phn_ids, phn_durations


def map_to_pred(batch, pool, processor, model, device='cuda'):
    inputs = processor(batch, sampling_rate=16_000, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
    
    return transcription

def _calc_wer(hyp_transcript, tgt_transcript):

    wer_errors = []

    for hyp in hyp_transcript:
        t = remove_specific_special_characters(tgt_transcript.upper())
        h = remove_specific_special_characters(hyp.upper())
        error = wer(t, h)
        wer_errors.append(error)

    return wer_errors

def calc_wer(eval_filelist_fp, audio_root_path, textgrid_root_path, audio_cfg = "./configs/audio.yaml", use_lm = False, device = "cuda"):
    
    audio_cfg = OmegaConf.load(audio_cfg)
    asr_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
    if use_lm :
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(
            labels=list(sorted_vocab_dict.keys()),
            kenlm_model_path="./checkpoints/4-gram.arpa.gz"
        )
        asr_processor = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=decoder
        )
    else:
        asr_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

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
    
    with open(eval_filelist_fp, "r") as f:
        eval_filelist = json.load(f)

    tq = tqdm(eval_filelist)
    wers = []

    for idx, filenames in enumerate(tq) :
        src_data, tgt_data = filenames
        
        try : 
            src_audio_fp, src_textgrid_fp = get_audio_textgrid_path(src_data, audio_root_path, textgrid_root_path)
        except :
            continue

        # obtain target transcript

        _, tgt_textgrid_fp = get_audio_textgrid_path(tgt_data, audio_root_path, textgrid_root_path)
        _, tgt_transcript = get_phoneme_frome_textgrid(tgt_textgrid_fp, return_transcript=True)


        tgt_audios = []
        n = 3
        for i in range(n) :
            tgt_audio_fp, tgt_textgrid_fp = src_audio_fp.replace("src",f"{i}"), src_textgrid_fp
            if os.path.exists(tgt_audio_fp) :
                tgt_mel, tgt_phn_expanded, tgt_phn_ids, tgt_phn_durations = process_prompt(to_mel, tgt_audio_fp, tgt_textgrid_fp, device)

        gt_wav = torch.from_numpy(sample_audio(src_audio_fp)).float().unsqueeze(0)
        gen_wav = torch.from_numpy(sample_audio(tgt_audio_fp)).float().unsqueeze(0)

        gen_wav_np = [wav for wav in gen_wav.squeeze(0).cpu().numpy()]
            
        with torch.no_grad():
            input_values = asr_processor([gen_wav_np], sampling_rate=16000, return_tensors="pt").input_values  # Batch size 1
            input_values = input_values.squeeze(0).to(device)
            logits = asr_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        
        if use_lm:
            hyp_transcript = []
            with get_context('fork').Pool(processes=4) as pool :
                lm_results = []
                for i in range(0, len(gen_wav_np), 4):
                    lm_results = map_to_pred(gen_wav_np[i : i + 4], pool, asr_processor, asr_model, device)
                    hyp_transcript.extend(lm_results)
        else:
            hyp_transcript = asr_processor.batch_decode(predicted_ids)

        wer_errors = _calc_wer(hyp_transcript, tgt_transcript)

        n_select = 1
        wer_indices = heapq.nsmallest(n_select, range(len(wer_errors)), key=wer_errors.__getitem__)
        # wer_indices = [0]
        word_error = [wer_errors[i] for i in wer_indices]
        word_error = np.mean(np.array(word_error))

        wers.append(word_error)

        tq.set_postfix({
            "WER" : np.mean(np.array(wers)),
        })
    
    return np.mean(np.array(wers))


