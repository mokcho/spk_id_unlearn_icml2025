import torch
import torch.nn.functional as F

def speaker_sim(emb1, emb2):

    return F.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2), dim=1)
