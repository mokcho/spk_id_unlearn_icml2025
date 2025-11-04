
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import defaultdict


def embed_per_spk(path, mode='default') : 
    spk_embs = defaultdict(list)

    for embed_file in os.listdir(path) :
        if embed_file.endswith(".npy") : 
            file_name = "_".join(embed_file.split("_")[:-1])
            if "src" in embed_file :
                if mode == 'src' :
                    spk_embs[file_name]= embed_file
                else :
                    continue
            else :
                if mode == 'default' :
                    spk_embs[file_name] = embed_file
                else :
                    continue
            
    spk_embs = dict(sorted(spk_embs.items()))
    return spk_embs

def load_embs(embs, dir) :
    embedding_list = None
    for file, file_paths in embs.items() :
        tmp = np.load(os.path.join(dir, file_paths)) #for file in file_paths]
        if tmp.ndim == 1:
            tmp = tmp.reshape(1, -1)  # Reshape to (1, 256)
        if embedding_list is None :
            embedding_list = np.vstack(tmp)
        else :
            embedding_list = np.vstack((embedding_list, tmp))
    return embedding_list

    
def process_embeddings(args):
    """
    Process all embeddings from both models and calculate the JS divergence.
    """
    unlearn_path = args.unlearn_path
    dumb_path = args.random_path
    
    src_path = unlearn_path
    results = defaultdict(list)
    
    # Get paths of each model
    unlearn_emb_paths = embed_per_spk(unlearn_path)
    dumb_emb_paths = embed_per_spk(dumb_path)
    
    if args.SIM :
        src_emb_paths = embed_per_spk(unlearn_path, mode='src')

    unlearn_embs = load_embs(unlearn_emb_paths, unlearn_path)
    dumb_embs = load_embs(dumb_emb_paths, dumb_path)
    if args.SIM :
        src_embs = load_embs(src_emb_paths, src_path)

    softmax_unlearn = softmax(normalize_embeddings(unlearn_embs), axis=1)
    softmax_dumb = softmax(normalize_embeddings(dumb_embs), axis=1)

    assert unlearn_embs.shape == softmax_unlearn.shape
        
    softmax_map = {}
    
    for i, (file_name, file_paths) in enumerate(unlearn_emb_paths.items()) :
        softmax_map[os.path.join(unlearn_path, file_name)] = softmax_unlearn[i, : ]

    for i, (file_name, file_paths) in enumerate(dumb_emb_paths.items()) :
        softmax_map[os.path.join(dumb_path, file_name)] = softmax_dumb[i, : ]
    
    if args.spkZRF :
    
        js_divergences = []
        for emb_name, emb_file in tqdm(unlearn_emb_paths.items()):
            
            assert emb_file.endswith(".npy")

            if os.path.join(dumb_path, emb_name) not in softmax_map :
                print(f"cannot find {emb_name} in dumb embeddings")

            
            dumb_emb_prob = softmax_map[os.path.join(dumb_path, emb_name)]
            unlearned_emb_prob = softmax_map[os.path.join(unlearn_path, emb_name)]

            assert np.array_equal(unlearned_emb_prob, dumb_emb_prob) == True, f"{np.array_equal(unlearned_emb_prob, dumb_emb_prob)}"
        
            # Calculate the JS divergence for each corresponding embedding
            unlearn_dumb_js = js_divergence(dumb_emb_prob, unlearned_emb_prob)
            js_divergences.append(unlearn_dumb_js)

            # Store the results
            results[emb_file] = {
                'js_divergence': unlearn_dumb_js
            }


    if args.spkZRF :
        mean_js = np.mean(js_divergences)
    
    if args.SIM :
        src_sim = np.array(speaker_sim(unlearn_embs, src_embs))
    
    return results, mean_js, src_sim
        
        

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearn_path", type=str, default='./outputs/forget/TGU/spk_embs', help="Path to the generated audio's speaker embeddings to be evaluated, e.g., './outputs/forget/CHECKPOINT/spk_embs'")
    parser.add_argument("--random_path", type=str, default='./outputs/forget/TGU/spk_embs', help="Path to the random generated audio's speaker embeddings, e.g., './outputs/forget/Voicebox_Random/spk_embs'")
    
    parser.add_argument("--spkZRF", action="store_true")
    parser.add_argument("--SIM", action="store_true")
    parser.add_argument("--WER", action="store_true")
    
    parser.add_argument("--eval_filelist_fp", type=str, default="./filelists/libriheavy_forget_eval_pair.json", required=False, help="Path to the evaluation filelist json, e.g., './filelists/eval_filelist.json'")
    parser.add_argument("--textgrid_root_path", type=str, default="/data/libriheavy/TextGrid/train/large/", required=False, help="Root path to the evaluation TextGrid files, e.g., './data/eval/textgrid'")    
    parser.add_argument("--audio_cfg", type=str, default="./configs/audio.yaml", help="Path to the audio config file." )
    
    
    args = parser.parse_args()

    
    print("#####################")
    print(f"Evaluating {args.unlearn_path} ")
    
    if args.spkZRF or args.SIM : 
        
        from eval.spkZRF import *
        from eval.sim import *

        results, zrf, sim  = process_embeddings(args)
        zrf_vanilla = zrf
        print(f"Mean spk-ZRF random <-> unlearned : {1-(np.nanmean(zrf))}")
        print(f"Mean speaker similarity : {np.nanmean(sim)}")
    
    if args.WER :
        from eval.wer import *
        
        # check for CSV file
        csv_path = os.path.join(os.path.dirname(args.unlearn_path), "output.csv")

        if os.path.exists(csv_path) :
            # Load the CSV
            df = pd.read_csv(csv_path)

            # Check if the columns exist
            for col in ['Similarity', 'WordError']:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in CSV.")

            # Compute means
            mean_similarity = df['Similarity'].mean()
            mean_worderror = df['WordError'].mean()

            print(f"Mean Similarity from CSV: {mean_similarity:.4f}")
            print(f"Mean WordError from CSV:  {mean_worderror:.4f}")
                
            # if no CSV file, use audio
        else : 
            
            raise NotImplementedError

            assert args.eval_filelist_fp is not None, "Please provide eval_filelist_fp for WER calculation."
            assert args.textgrid_root_path is not None, "Please provide textgrid_root_path for WER calculation."

            mean_worderror = calc_wer(args.eval_filelist_fp, os.path.join(os.path.dirname(args.unlearn_path),"wavs" ), args.textgrid_root_path, audio_cfg = args.audio_cfg, use_lm = False)
            print(f"Mean WER : {mean_worderror}")
            
        
    
    print("#####################")
