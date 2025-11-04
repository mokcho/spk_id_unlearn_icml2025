import os
import math
import numpy as np
import textgrid
import json
import torch
import torchaudio


def cut_by_vad(wav, sr):
    vad_front = torchaudio.functional.vad(wav, sample_rate=sr)
    vad_front_reversed = torch.flip(vad_front, dims=[0])
    vad_back = torchaudio.functional.vad(vad_front_reversed, sample_rate=sr)
    vad_result = torch.flip(vad_back, dims=[0])
    return vad_result


def read_json_file(path):
    with open(path, "r") as f :
        return json.load(f)


def read_npy(path):
    try :
        data = np.load(path)
    except :
        print(f"Error occuredd while reading : {path}")
    return data


def read_txt_file(filepath):
    with open(filepath, "r", encoding='utf-8') as f:
        line = f.readline()
    line = line.strip()
    return line


def get_phoneme_frome_textgrid(filepath, return_transcript=False):
    tg = textgrid.TextGrid.fromFile(filepath)
    phns = tg[-1]
    phns = [[p.mark, p.minTime, p.maxTime] for p in phns]
    for i in range(len(phns)):
        if phns[i][0] == '':
            phns[i] = ['_', phns[i][1], phns[i][2]]

    # Add Space segments
    words = tg[0]
    words = [(w.mark, w.minTime, w.maxTime) for w in words]
    for i in range(len(words)):
        if words[i][0] == '':
            words[i] = ['_', words[i][1], words[i][2]]

    word_end_times = [w[-1] for w in words]

    _phns = []
    for i in range(len(phns)):
        end_time = phns[i][-1]
        _phns.append(phns[i])
        if i < (len(phns)-1):
            if (end_time in word_end_times) and _phns[-1][0] != "_" : #(phns[i+1][0] != "_") :
                _phns.append(["_", end_time, end_time])
    
    if len(_phns) > 2:
        if _phns[-1][0] == _phns[-2][0]:
            _phns[-2][-1] = _phns[-1][-1]
            _phns = _phns[:-1]
    
    phns = add_pos_symbol(_phns)

    # Deletec duplicated space
    _phns = [phns[0]]
    for i in range(1, len(phns)):
        if phns[i][0] != '_' :
            _phns.append(phns[i])
            continue
        else:
            if _phns[-1][0] != '_':
                _phns.append(phns[i])
                continue
            else:
                _phns[-1][-1] = phns[i][-1]
    phns = _phns

    if return_transcript :
        space_tokens = ['_', " "]
        transcript = [w[0] + ' ' if w[0] not in space_tokens else '' for w in words]
        transcript = "".join(transcript)
        return phns, transcript

    return phns

def add_pos_symbol(segments):
    # Add position {S, M, E} to each phonemes
    _segments = []
    for i, seg in enumerate(segments):
        seg = list(seg)
        # In case of the first segment
        if (len(_segments) == 0) :
            if seg[0] == '_': # For the spaces
                _segments.append(seg)
                continue
            else:
                seg[0] = f'S_{seg[0]}'
                _segments.append(seg)
                continue
        
        # In case of the first phoneme of the words
        if _segments[-1][0] == '_' and seg[0] != '_':
            seg[0] = f'S_{seg[0]}'
            _segments.append(seg)
            continue
        
        if seg[0] == '_': # In Case of the space
            if not _segments[-1][0].startswith('S') : # End of the words
                _segments[-1][0] = _segments[-1][0].replace("M", "E")
            _segments.append(seg)
        else:
            # In case of the middle phoneme
            seg[0] = f'M_{seg[0]}'
            _segments.append(seg)

    if _segments[-1][0] != '_' and (not _segments[-1][0].startswith('S')):
        _segments[-1][0] = _segments[-1][0].replace("M", "E")
    
    return _segments


def get_duration(segments, num_frames, ratio):
    seg_dur = []
    start_frame = 0
    for i, p in enumerate(segments):
        if i == len(segments) - 1: # last phoneme
            if num_frames is not None:
                end_frame = num_frames
            else:
                end_frame = math.ceil(p[2] * ratio)
        else:
            end_frame = math.ceil(p[2] * ratio)
        
        n_frame = end_frame - start_frame
        start_frame = end_frame
            
        seg_dur.append(n_frame)
    
    if num_frames is not None:
        assert num_frames == sum(seg_dur)
    
    return np.array(seg_dur)


def make_base_dirs(file_path):
    base_dirs = file_path.split("/")[:-1]
    base_dirs = "/" + "/".join(base_dirs)
    os.makedirs(base_dirs, exist_ok=True)