import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_llamav_embeddings(data_root, device):
    json_path = os.path.join(data_root, "llamav_info.json")
    with open(json_path, "r") as f:
        info = json.load(f)
    
    all_embeddings = []
    all_image_pths = []
    for image_info in tqdm(info["images"], desc="Loading Llamav embeddings"):
        emb_path = os.path.join(data_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
        embeddings = torch.load(emb_path, map_location=device)
        pixel_embeds = embeddings["pixel_embeds"].half()
        h,w,c = pixel_embeds.shape
        pixel_embeds = F.interpolate(pixel_embeds.permute(2,0,1).unsqueeze(0), size=(h//2, w//2), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
        all_embeddings.append(pixel_embeds.cpu())
        all_image_pths.append(image_info["file_name"])
    
    all_embeddings = torch.stack(all_embeddings, dim=0)
    return all_embeddings, all_image_pths

def unravel_indices(flat_indices, shape):
    """
    Convert flattened indices into unraveled indices corresponding to a given shape.

    Args:
        flat_indices (torch.Tensor): A 1D tensor of flattened indices.
        shape (tuple): The shape (n, h, w) of the original tensor.

    Returns:
        tuple of torch.Tensors: The unraveled indices for each dimension (n_indices, h_indices, w_indices).
    """
    n, h, w = shape
    n_indices = flat_indices // (h * w)
    hw_remainder = flat_indices % (h * w)
    h_indices = hw_remainder // w
    w_indices = hw_remainder % w
    return torch.stack((n_indices, h_indices, w_indices)).t()

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/3dgs/garden"  # Update this path as needed
    
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading LLaMAv embeddings...")
    all_embeddings, all_image_pths = load_llamav_embeddings(data_root, device)

    # 0.60, 0.65
    threshold = 0.65
    mem_path = os.path.join(data_root, "llamav", f"mem{int(threshold*100)}.emb")
    mem_embeddings = torch.load(mem_path, map_location=device)

    all_embeddings = all_embeddings.cuda().float()
    mem_embeddings = mem_embeddings.cuda().float()
    
    all_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
    mem_embeddings = mem_embeddings / mem_embeddings.norm(dim=1, keepdim=True)
    
    info_mem = {}
    for idx, mem_embed in enumerate(tqdm(mem_embeddings, desc="Processing embeddings")):
        similarity = all_embeddings.cuda() @ mem_embed.cuda()
        n,h,w = similarity.shape
        view_similarity = similarity.view(-1)
        topk_indices = torch.topk(view_similarity, k=5, dim=0).indices
        indices_nhw = unravel_indices(topk_indices, (n, h, w)).float()
        indices_nhw[:, 1] = indices_nhw[:, 1] * 1.0 / h
        indices_nhw[:, 2] = indices_nhw[:, 2] * 1.0 / w
        
        topk_list = [{"image_name": all_image_pths[int(x)], "point_height": h.item(), "point_width": w.item()} for x,h,w in indices_nhw]
        info_mem[idx] = topk_list

    mem2fea_path = os.path.join(data_root, "llamav", f"mem{int(threshold*100)}.index")
    torch.save(info_mem, mem2fea_path)