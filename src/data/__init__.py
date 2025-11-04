
def collate_fn(batch):
    keys = batch[0].keys()
    new_dict = {}
    for k in keys:
        new_dict[k] = []
    
    for b in batch:
        for k in b.keys():
            new_dict[k].append(b[k])
    
    return new_dict