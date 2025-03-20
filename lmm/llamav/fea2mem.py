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
    for image_info in tqdm(info["images"], desc="Loading Llamav embeddings"):
        emb_path = os.path.join(data_root, "/".join(image_info["emb_pth"].split('/')[-3:]))
        embeddings = torch.load(emb_path, map_location=device)
        pixel_embeds = embeddings["pixel_embeds"].half()
        h,w,c = pixel_embeds.shape
        pixel_embeds = F.interpolate(pixel_embeds.permute(2,0,1).unsqueeze(0), size=(h//2, w//2), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
        pixel_embeds = pixel_embeds.view(-1, pixel_embeds.shape[-1])
        all_embeddings.append(pixel_embeds.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# def filter_embeddings_large_scale(embeddings, threshold=0.9, chunk_size=1000, similarity_batch_size=800000):
#     num_embeddings = embeddings.shape[0]
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Normalize the embeddings
#     normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
#     filtered_indices = []
#     used_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)
    
#     for i in tqdm(range(0, num_embeddings, chunk_size), desc="Filtering embeddings"):
#         chunk = normalized_embeddings[i:i+chunk_size].to(device)
        
#         # Compute similarity of this chunk with all embeddings in batches
#         batched_chunk = []
#         for j in range(0, num_embeddings, similarity_batch_size):
#             similarity_batch = normalized_embeddings[j:j+similarity_batch_size].to(device)
#             similarity_chunk = torch.mm(chunk, similarity_batch.t())
#             batched_chunk += [similarity_chunk.cpu()]
            
#             del similarity_batch
#             torch.cuda.empty_cache()
        
#         batched_chunk = torch.cat(batched_chunk, dim=1)
#         for k in range(chunk.shape[0]):
#             if used_mask[i+k]:
#                 continue
            
#             similar_indices = torch.where(batched_chunk[k] >= threshold)[0]
#             if used_mask[similar_indices].sum() < len(similar_indices):
#                 filtered_indices.append(i+k)
#                 used_mask[similar_indices] = True

#         del chunk
#         torch.cuda.empty_cache()

#     filtered_embeddings = embeddings[filtered_indices]
#     return filtered_embeddings

def filter_embeddings_large_scale(embeddings, threshold=0.9, chunk_size=1000, similarity_batch_size=800000):
    num_embeddings = embeddings.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize the embeddings on CPU to save GPU memory
    embeddings = embeddings.float()
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    filtered_indices = []
    used_mask = torch.zeros(num_embeddings, dtype=torch.bool, device='cpu')
    
    for i in tqdm(range(0, num_embeddings, chunk_size), desc="Filtering embeddings"):
        chunk = normalized_embeddings[i:i+chunk_size].to(device)
        
        # Compute similarity of this chunk with all embeddings in batches
        similarity_scores = torch.zeros(chunk.shape[0], num_embeddings, device='cpu')
        for j in range(0, num_embeddings, similarity_batch_size):
            similarity_batch = normalized_embeddings[j:j+similarity_batch_size].to(device)
            similarity_chunk = torch.mm(chunk, similarity_batch.t())
            similarity_scores[:, j:j+similarity_batch_size] = similarity_chunk.cpu()
            del similarity_batch
        
        # Process the chunk
        for k in range(chunk.shape[0]):
            if not used_mask[i+k]:
                similar_indices = torch.where(similarity_scores[k] >= threshold)[0]
                if not used_mask[similar_indices].any():
                    filtered_indices.append(i+k)
                    used_mask[similar_indices] = True
        
        del chunk, similarity_scores
        torch.cuda.empty_cache()
    return embeddings[filtered_indices]

if __name__ == "__main__":
    data_root = "/data/xueyanz/data/3dgs/train"  # Update this path as needed
    
    # Set up CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading LLaMAv embeddings...")
    all_embeddings = load_llamav_embeddings(data_root, device)
    print(f"Loaded embeddings shape: {all_embeddings.shape}")
    
    print("Filtering embeddings...")
    # 0.60, 0.65
    threshold = 0.65
    mem_embeddings = filter_embeddings_large_scale(all_embeddings, threshold=threshold, chunk_size=1000)
    print(f"Filtered embeddings shape: {mem_embeddings.shape}")
    
    # Save the filtered embeddings
    output_path = os.path.join(data_root, "llamav", f"mem{int(threshold*100)}.emb")
    torch.save(mem_embeddings.cpu(), output_path)
    print(f"Saved filtered embeddings to {output_path}")