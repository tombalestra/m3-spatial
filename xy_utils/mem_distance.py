import torch
import os

data_name = "train"
model_types = ["clip", "dinov2", "llama3", "llamav", "seem", "siglip"]
mem_names = ["mem80.emb", "mem75.emb", "mem100.emb", "mem60.emb", "mem100.emb", "mem85.emb"]
data_root = "/disk1/data/m3/data_v2/{}".format(data_name)

def compute_distances_and_stats(embeddings, num_pairs=1000):
    """
    Compute distances and feature scale using 1-99 percentile range of L2 norms.
    Also checks if embeddings appear to be L2 normalized.
    
    Args:
        embeddings: torch.Tensor of shape [N, D] where N is number of embeddings and D is dimension
        num_pairs: int, number of random pairs to sample for distances
    
    Returns:
        dict: containing distances, L2 norm scale, and normalization status
    """
    num_embeddings = embeddings.shape[0]
    
    # Check if embeddings are L2 normalized
    embedding_norms = torch.norm(embeddings, p=2, dim=1)  # Shape: [N]
    mean_norm = torch.mean(embedding_norms).item()
    std_norm = torch.std(embedding_norms).item()
    
    # If vectors are L2 normalized, their norms should be very close to 1.0
    # We use a threshold to determine if they're normalized
    is_normalized = (abs(mean_norm - 1.0) < 0.1) and (std_norm < 0.1)
    
    # Generate random pairs of indices for distances
    idx1 = torch.randint(0, num_embeddings, (num_pairs,))
    idx2 = torch.randint(0, num_embeddings, (num_pairs,))
    
    # Get the pairs of embeddings
    pairs1 = embeddings[idx1]
    pairs2 = embeddings[idx2]
    
    # Compute regular L2 distances between pairs
    distances = torch.norm(pairs1 - pairs2, p=2, dim=1)
    avg_distance = torch.mean(distances).item()
    
    # Compute normalized L2 distances
    pairs1_normalized = pairs1 / torch.norm(pairs1, p=2, dim=1, keepdim=True)
    pairs2_normalized = pairs2 / torch.norm(pairs2, p=2, dim=1, keepdim=True)
    normalized_distances = torch.norm(pairs1_normalized - pairs2_normalized, p=2, dim=1)
    avg_normalized_distance = torch.mean(normalized_distances).item()
    
    # Compute 1-99 percentile scale of the norms
    percentiles = torch.tensor([1, 99], dtype=torch.float)
    p1, p99 = torch.quantile(embedding_norms, percentiles/100)
    norm_scale = p99 - p1
    
    return {
        'avg_l2': avg_distance,
        'avg_normalized_l2': avg_normalized_distance,
        'norm_p1': p1.item(),
        'norm_p99': p99.item(),
        'norm_scale': norm_scale.item(),
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'is_normalized': is_normalized
    }

# Process each model's embeddings
for model_type, mem_name in zip(model_types, mem_names):
    mem_root = os.path.join(data_root, model_type, mem_name)
    mem_embed = torch.load(mem_root).float()
    
    # Compute metrics
    metrics = compute_distances_and_stats(mem_embed)
    
    # Print results
    print(f"\nModel: {model_type}")
    print(f"L2 Normalized: {'Yes' if metrics['is_normalized'] else 'No'}")
    print(f"Mean vector norm: {metrics['mean_norm']:.4f} (std: {metrics['std_norm']:.4f})")
    print(f"Average L2 distance: {metrics['avg_l2']:.4f}")
    print(f"Average normalized L2 distance: {metrics['avg_normalized_l2']:.4f}")
    print(f"L2 norm scale (p99 - p1): {metrics['norm_scale']:.4f}")
    print(f"  [p1: {metrics['norm_p1']:.4f}, p99: {metrics['norm_p99']:.4f}]")