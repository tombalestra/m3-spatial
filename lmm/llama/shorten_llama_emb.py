import os
import glob
import torch


emb_path = "/disk1/data/m3/data_v2/tabletop_v2/llama3/raw_embeds"
emb_files = glob.glob(os.path.join(emb_path, "*.emb"))

out_path = emb_path.replace("raw_embeds", "embeds")
os.makedirs(out_path, exist_ok=True)

for emb_file in emb_files:
    print(emb_file)
    embeddings = torch.load(emb_file, map_location='cpu')
    emb_dict = {}
    for key, value in embeddings.items():
        if 'description' in value:
            emb_dict[key] = value['description'][-12].mean(dim=0).cpu()
        elif 'short description' in value:
            emb_dict[key] = value['short description'][-12].mean(dim=0).cpu()
    torch.save(emb_dict, os.path.join(out_path, os.path.basename(emb_file)))