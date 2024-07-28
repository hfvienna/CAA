import torch
import matplotlib.pyplot as plt
from gemma_1_wrapper import Gemma1Wrapper
from generate_vectors import generate_save_vectors_for_behavior
from sae_lens import SAE
import os
from behaviors import get_vector_path
from behaviors import COORDINATE, ALL_BEHAVIORS, get_vector_path
from behaviors import BASE_DIR, COORDINATE, ALL_BEHAVIORS, get_vector_path

HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

def decompose_caa_vector(sae, caa_vector, top_k=20):
    caa_vector = caa_vector.to(next(sae.parameters()).device).view(1, -1)
    
    with torch.no_grad():
        sparse_features = sae.encode(caa_vector)
    
    top_values, top_indices = torch.topk(sparse_features.abs().squeeze(), k=top_k)
    
    top_features = {idx.item(): val.item() for idx, val in zip(top_indices, top_values)}
    
    feature_contributions = {}
    for idx, activation in top_features.items():
        feature_vector = sae.W_dec[idx]
        contribution = activation * feature_vector
        feature_contributions[idx] = contribution
    
    sparse_reconstruction = torch.zeros_like(sparse_features)
    sparse_reconstruction[0, top_indices] = sparse_features[0, top_indices]
    with torch.no_grad():
        reconstructed = sae.decode(sparse_reconstruction)
    
    reconstruction_error = torch.nn.functional.mse_loss(caa_vector, reconstructed)
    
    return top_features, feature_contributions, reconstructed.squeeze(), reconstruction_error.item()

def analyze_gemma_vector(behavior, layer=10, model_name_path='gemma-2b'):
    vector_path = get_vector_path(behavior, layer, model_name_path)
    caa_vector = torch.load(vector_path)
    
    print(f"Analyzing vector for behavior: {behavior}")
    print(f"Vector shape: {caa_vector.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move caa_vector to CUDA
    caa_vector = caa_vector.to(device)
    
    # Load the SAE model
    sae_layer_use = layer
    sae, cfg_dict, _ = SAE.from_pretrained(
        release="gemma-2b-res-jb",
        sae_id=f"blocks.{sae_layer_use}.hook_resid_post"
    )
    sae = sae.to(device)
    
    # Get hook point
    hook_point = sae.cfg.hook_name
    print(f"Hook point: {hook_point}")
    
    # Decompose the CAA vector
    top_features, feature_contributions, reconstructed, error = decompose_caa_vector(sae, caa_vector)
    
    # Print results
    print("Top SAE features for CAA vector:")
    for rank, (idx, activation) in enumerate(top_features.items(), 1):
        contribution = feature_contributions[idx].to(device)
        contribution_magnitude = torch.norm(contribution).item()
        contribution_direction = torch.cosine_similarity(caa_vector.squeeze(), contribution.squeeze(), dim=0).item()
        print(f"Rank {rank}: Feature {idx}")
        print(f"  Activation: {activation:.4f}")
        print(f"  Contribution Magnitude: {contribution_magnitude:.4f}")
        print(f"  Contribution Direction (cosine similarity): {contribution_direction:.4f}")
    
    print(f"\nReconstruction error: {error:.6f}")
    
    # Visualize the reconstruction and top feature contributions
    plt.figure(figsize=(12, 6))
    plt.plot(caa_vector.cpu().detach().numpy().squeeze(), label='Original CAA Vector', alpha=0.7)
    plt.plot(reconstructed.cpu().detach().numpy(), label='Reconstructed from SAE', alpha=0.7)
    
    # Plot top 5 feature contributions
    for i, (idx, contribution) in enumerate(list(feature_contributions.items())[:5]):
        plt.plot(contribution.cpu().detach().numpy(), label=f'Feature {idx} Contribution', alpha=0.5)
    
    plt.legend()
    plt.title(f'CAA Vector, Reconstruction, and Top Feature Contributions for {behavior}')
    plt.savefig(f'{behavior}_reconstruction.png')
    plt.close()
    
    # Visualize cumulative explained variance
    cumulative_contribution = torch.zeros_like(caa_vector.squeeze())
    explained_variances = []
    for contribution in feature_contributions.values():
        cumulative_contribution += contribution
        explained_variance = 1 - torch.nn.functional.mse_loss(cumulative_contribution, caa_vector.squeeze()) / torch.var(caa_vector.squeeze())
        explained_variances.append(explained_variance.item())
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variances) + 1), explained_variances)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Cumulative Explained Variance by Top SAE Features for {behavior}')
    plt.savefig(f'{behavior}_explained_variance.png')
    plt.close()
    
    print(f"Analysis complete for {behavior}")
    print(f"Hook point: {hook_point}")


if __name__ == "__main__":
    import sys
    from behaviors import COORDINATE, ALL_BEHAVIORS
    if len(sys.argv) > 1 and sys.argv[1] in ALL_BEHAVIORS:
        behavior = sys.argv[1]
    else:
        behavior = COORDINATE
    analyze_gemma_vector(behavior, layer=10, model_name_path='gemma-2b')