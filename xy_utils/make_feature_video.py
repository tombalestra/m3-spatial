import glob
import torch
import torch.nn.functional as F

from xy_utils.visual import vpca_embeddings, compute_global_pca
import cv2


embed_paths = sorted(glob.glob("/disk1/data/m3/data_v2/geisel/siglip/embeds/*.emb"))
output_path = "/home/xueyan/output/mmm/geisel/siglip_raw"

pca_components = None
for embed_path in embed_paths:
    embeddings = torch.load(embed_path)['pixel_embeds']
    embeddings = F.interpolate(embeddings.permute(2,0,1)[None,], size=(27, 48), mode='bilinear', align_corners=False)[0]
    if pca_components is None:
        pca_components = compute_global_pca(embeddings)
    image = vpca_embeddings(embeddings.cpu(), **pca_components)
    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{output_path}/{embed_path.split('/')[-1].replace('.emb', '.png')}", image)
    print(f"Saved {embed_path.split('/')[-1].replace('.emb', '.png')}")
