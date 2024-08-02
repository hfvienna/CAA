import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch

from behaviors import ALL_BEHAVIORS, BASE_DIR, COORDINATE, get_vector_path
from gemma_2_wrapper import Gemma2Wrapper
from generate_vectors import generate_save_vectors_for_behavior
from huggingface_hub import hf_hub_download
import numpy as np
import torch.nn as nn

HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
TOP_K_FEATURES = 20

def decompose_caa_vector(sae, caa_vector, top_k=20):
    caa_vector = caa_vector.to(sae.W_enc.device).view(1, -1)
    
    with torch.no_grad():
        sparse_features = sae.encode(caa_vector)
    
    top_features = dict(sorted(
        {i: v.item() for i, v in enumerate(sparse_features.squeeze())}.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k])
    
    feature_contributions = {}
    for idx, activation in top_features.items():
        feature_vector = sae.W_dec[idx]
        contribution = activation * feature_vector
        feature_contributions[idx] = contribution
    
    sparse_reconstruction = torch.zeros_like(sparse_features)
    top_indices = list(top_features.keys())
    sparse_reconstruction[0, top_indices] = sparse_features[0, top_indices]
    with torch.no_grad():
        reconstructed = sae.decode(sparse_reconstruction)
    
    reconstruction_error = torch.nn.functional.mse_loss(caa_vector, reconstructed)
    
    return top_features, feature_contributions, reconstructed.squeeze(), reconstruction_error.item()

def generate_behavior_curve(model, vector, multipliers=[-1, -0.5, 0, 0.5, 1]):
    results = []
    for multiplier in multipliers:
        scaled_vector = vector * multiplier
        output = model.get_behavior_from_vector(scaled_vector)
        # Extract a numerical value from the output
        value = extract_numerical_value(output)
        results.append(value)
    return results

def extract_numerical_value(output):
    # The output is already a numerical value (average log probability)
    return output

def fix_vector_path(path):
    parts = path.split('_')
    if 'feature' in parts:
        feature = parts[parts.index('feature') + 1]
        layer = parts[parts.index('layer') + 1]
        model = '_'.join(parts[parts.index('gemma'):])
        path = os.path.join(os.path.dirname(path), f"feature_{feature}_layer_{layer}_{model}.pt")
    elif 'vec' in parts:
        layer = parts[parts.index('layer') + 1]
        model = '_'.join(parts[parts.index('gemma'):])
        path = os.path.join(os.path.dirname(path), f"vec_layer_{layer}_{model}.pt")
    
    # Remove duplicate '.pt' extension if present
    while path.endswith('.pt.pt'):
        path = path[:-3]
    
    return path

def analyze_gemma_vector(behavior, layer=10, model_name_path='gemma-2b', only_combined_vector=False):
    global TOP_K_FEATURES
    vector_path = fix_vector_path(get_vector_path(behavior, layer, model_name_path))
    normalized_dir = os.path.join(BASE_DIR, 'normalized_vectors', behavior)

    # Load the existing normalized vector
    original_path = os.path.join(normalized_dir, f"vec_layer_{layer}_{model_name_path}.pt")
    original_path = fix_vector_path(original_path)
    caa_vector = torch.load(original_path)

    print(f"Analyzing vector for behavior: {behavior}")
    print(f"Vector shape: {caa_vector.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move caa_vector to device
    caa_vector = caa_vector.to(device)

    print(f"Norm of the original vector: {caa_vector.norm().item():.4f}")

    # Load the SAE model
    sae_model_name = "google/gemma-scope-2b-pt-res"
    path_to_params = hf_hub_download(
        repo_id=sae_model_name,
        filename=f"layer_{layer}/width_16k/average_l0_83/params.npz",
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    class JumpReLUSAE(nn.Module):
        def __init__(self, d_model, d_sae):
            super().__init__()
            self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
            self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
            self.threshold = nn.Parameter(torch.zeros(d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.b_dec = nn.Parameter(torch.zeros(d_model))

        def encode(self, input_acts):
            pre_acts = input_acts @ self.W_enc + self.b_enc
            mask = (pre_acts > self.threshold)
            acts = mask * torch.nn.functional.relu(pre_acts)
            return acts

        def decode(self, acts):
            return acts @ self.W_dec + self.b_dec

        def forward(self, acts):
            acts = self.encode(acts)
            recon = self.decode(acts)
            return recon

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)
    sae = sae.to(device)

    # Get the hook point
    hook_point = f"blocks.{layer}.hook_resid_post"
    print(f"Hook point: {hook_point}")

    # Calculate and print the norm for all features
    all_features_norm = torch.norm(sae.W_dec, dim=1)
    print(f"Norm of all features: {all_features_norm.mean().item():.4f} (mean), {all_features_norm.min().item():.4f} (min), {all_features_norm.max().item():.4f} (max)")

    # Decompose the CAA vector
    top_features, feature_contributions, reconstructed, error = decompose_caa_vector(sae, caa_vector)

    if only_combined_vector:
        # Combine top features
        combined_vector = torch.zeros_like(caa_vector)
        for idx in list(top_features.keys())[:TOP_K_FEATURES]:
            combined_vector += feature_contributions[idx]

        # Save the combined vector
        combined_vector_path = os.path.join(normalized_dir, f"features_top{TOP_K_FEATURES}combined_layer_{layer}_{model_name_path}.pt")
        torch.save(combined_vector, combined_vector_path)
        print(f"Saved combined top {TOP_K_FEATURES} features vector to {combined_vector_path}")

        # Calculate and print the norm of the combined vector
        combined_norm = combined_vector.norm().item()
        print(f"Norm of combined top {TOP_K_FEATURES} features vector: {combined_norm:.4f}")

        # Calculate and print the cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(caa_vector.view(1, -1), combined_vector.view(1, -1)).item()
        print(f"Cosine similarity between original and combined vector: {cosine_similarity:.4f}")

        # Calculate reconstructions
        full_reconstruction = sae.decode(sae.encode(caa_vector))
        top_k_reconstruction = torch.zeros_like(caa_vector)
        for idx in list(top_features.keys())[:TOP_K_FEATURES]:
            top_k_reconstruction += feature_contributions[idx]

        # 1. Cosine Similarity
        full_cosine_similarity = torch.nn.functional.cosine_similarity(caa_vector.view(1, -1), full_reconstruction.view(1, -1)).item()
        top_k_cosine_similarity = torch.nn.functional.cosine_similarity(caa_vector.view(1, -1), top_k_reconstruction.view(1, -1)).item()
        print(f"Cosine similarity - Full reconstruction: {full_cosine_similarity:.4f}")
        print(f"Cosine similarity - Top {TOP_K_FEATURES} reconstruction: {top_k_cosine_similarity:.4f}")

        # 2. Norm Comparison
        original_norm = caa_vector.norm().item()
        full_reconstruction_norm = full_reconstruction.norm().item()
        top_k_reconstruction_norm = top_k_reconstruction.norm().item()
        print(f"Norm - Original: {original_norm:.4f}")
        print(f"Norm - Full reconstruction: {full_reconstruction_norm:.4f}")
        print(f"Norm - Top {TOP_K_FEATURES} reconstruction: {top_k_reconstruction_norm:.4f}")

        # 3. Mean Squared Error and Loss Recovered
        full_mse = torch.nn.functional.mse_loss(caa_vector, full_reconstruction).item()
        print(f"MSE - Full reconstruction: {full_mse:.4f}")

        # Calculate MSE and Loss Recovered for different numbers of top features
        feature_counts = [10, 20, 50, 100, 200, 500]
        for count in feature_counts:
            top_k_reconstruction = torch.zeros_like(caa_vector)
            for idx in list(top_features.keys())[:count]:
                top_k_reconstruction += feature_contributions[idx]
            top_k_mse = torch.nn.functional.mse_loss(caa_vector, top_k_reconstruction).item()
            loss_recovered_mse = calculate_loss_recovered(caa_vector, top_k_reconstruction, 'mse')
            loss_recovered_cosine = calculate_loss_recovered(caa_vector, top_k_reconstruction, 'cosine')
            print(f"Top {count} reconstruction:")
            print(f"  MSE: {top_k_mse:.4f}")
            print(f"  Loss Recovered (MSE): {loss_recovered_mse:.2f}%")
            print(f"  Loss Recovered (Cosine): {loss_recovered_cosine:.2f}%")

        # Calculate MSE and Loss Recovered for random subsets of features
        random_counts = [100, 500, 1000]
        for count in random_counts:
            random_reconstruction = torch.zeros_like(caa_vector)
            random_indices = random.sample(list(top_features.keys()), count)
            for idx in random_indices:
                random_reconstruction += feature_contributions[idx]
            random_mse = torch.nn.functional.mse_loss(caa_vector, random_reconstruction).item()
            loss_recovered_mse = calculate_loss_recovered(caa_vector, random_reconstruction, 'mse')
            loss_recovered_cosine = calculate_loss_recovered(caa_vector, random_reconstruction, 'cosine')
            print(f"Random {count} reconstruction:")
            print(f"  MSE: {random_mse:.4f}")
            print(f"  Loss Recovered (MSE): {loss_recovered_mse:.2f}%")
            print(f"  Loss Recovered (Cosine): {loss_recovered_cosine:.2f}%")

        # 4. Cross Entropy Loss
        model = Gemma2Wrapper(model_name_path)
        model = model.to(device)

        original_logits = model(caa_vector.unsqueeze(0))
        full_reconstruction_logits = model(full_reconstruction.unsqueeze(0))
        top_k_reconstruction_logits = model(top_k_reconstruction.unsqueeze(0))

        criterion = torch.nn.CrossEntropyLoss()
        target = torch.argmax(original_logits, dim=-1)

        original_loss = criterion(original_logits, target)
        full_reconstruction_loss = criterion(full_reconstruction_logits, target)
        top_k_reconstruction_loss = criterion(top_k_reconstruction_logits, target)

        print(f"Cross entropy loss - Original: {original_loss.item():.4f}")
        print(f"Cross entropy loss - Full reconstruction: {full_reconstruction_loss.item():.4f}")
        print(f"Cross entropy loss - Top {TOP_K_FEATURES} reconstruction: {top_k_reconstruction_loss.item():.4f}")

        # 5. Activation Distribution
        def print_activation_stats(tensor, name):
            print(f"{name} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}")

        print_activation_stats(caa_vector, "Original")
        print_activation_stats(full_reconstruction, "Full reconstruction")
        print_activation_stats(top_k_reconstruction, f"Top {TOP_K_FEATURES} reconstruction")

        # 6. Behavioral Analysis is already being done through steering analysis

        # Calculate reconstruction loss for different numbers of top features
        reconstruction_losses = []
        feature_counts = range(1, TOP_K_FEATURES + 1, 1)  # Adjust step size if needed
        for k in feature_counts:
            top_k_reconstruction = torch.zeros_like(caa_vector)
            for idx in list(top_features.keys())[:k]:
                top_k_reconstruction += feature_contributions[idx]
            loss = torch.nn.functional.mse_loss(caa_vector, top_k_reconstruction).item()
            reconstruction_losses.append(loss)

        # Plot reconstruction loss
        plt.figure(figsize=(10, 6))
        plt.plot(feature_counts, reconstruction_losses, marker='o')
        plt.title(f'Reconstruction Loss vs Number of Top Features for {behavior}')
        plt.xlabel('Number of Top Features')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.yscale('log')  # Use log scale for better visualization
        plt.grid(True)
        plot_path = os.path.join(normalized_dir, f"reconstruction_loss_plot_{layer}_{model_name_path}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Reconstruction loss plot saved to {plot_path}")

        original_norm = caa_vector.norm().item()
        feature_norms = {idx: feature_contributions[idx].norm().item() for idx in top_features.keys()}
        return top_features, feature_norms, original_norm

    # Save top feature vectors in normalized_vectors folder
    for idx in list(top_features.keys())[:TOP_K_FEATURES]:
        feature_vector = feature_contributions[idx]
        feature_path = os.path.join(normalized_dir, f"feature_{idx}_layer_{layer}_{model_name_path}.pt")
        feature_path = fix_vector_path(feature_path)
        torch.save(feature_vector, feature_path)

    print(f"Saved top {TOP_K_FEATURES} feature vectors in {normalized_dir}")
    print("To process these vectors, run:")
    print(f"python prompting_with_steering.py --layers $(seq {layer-1} {layer+1}) --multipliers -0.5 0 0.5 --type ab --model_size 2b --model_type gemma_1 --use_base_model")
    print("Then, to plot the results, run:")
    print(f"python plot_results.py --layers {layer} --multipliers -1 -0.5 0 0.5 1 --type ab --model_type gemma_1 --model_size 2b --use_base_model --behavior {behavior}")

    # Run steering analysis for the original vector and top feature vectors
    print(f"Running steering analysis for the original vector and top {TOP_K_FEATURES} feature vectors...")
    for idx in ['original'] + list(top_features.keys())[:TOP_K_FEATURES]:
        if idx == 'original':
            vector_path = original_path
        else:
            vector_path = os.path.join(normalized_dir, f"feature_{idx}_layer_{layer}_{model_name_path}.pt")
            vector_path = fix_vector_path(vector_path)
        command = f"python prompting_with_steering.py --layers $(seq {layer-1} {layer+1}) --multipliers -0.5 0 0.5 --type ab --model_size 2b --model_type gemma_1 --use_base_model --behavior {behavior} --override_vector_model {vector_path}"
        print(f"Running command: {command}")
        os.system(command)

    print("Steering analysis complete. Results saved in the results folder.")

    # Plot the steering results
    print("Plotting steering results...")
    command = f"python plot_results.py --layers {layer} --multipliers -1 -0.5 0 0.5 1 --type ab --model_type gemma_1 --model_size 2b --use_base_model --behavior {behavior}"
    print(f"Running command: {command}")
    os.system(command)

    print("Plotting complete. Results saved in the plots folder.")

    # Combine top features
    combined_vector = torch.zeros_like(caa_vector)
    for idx in list(top_features.keys())[:TOP_K_FEATURES]:
        combined_vector += feature_contributions[idx]

    # Save the combined vector
    combined_vector_path = os.path.join(normalized_dir, f"features_top{TOP_K_FEATURES}combined_layer_{layer}_{model_name_path}.pt")
    torch.save(combined_vector, combined_vector_path)
    print(f"Saved combined top {TOP_K_FEATURES} features vector to {combined_vector_path}")

    # Calculate and print the norm of the combined vector
    combined_norm = combined_vector.norm().item()
    print(f"Norm of combined top {TOP_K_FEATURES} features vector: {combined_norm:.4f}")

    # Calculate cross entropy loss
    model = Gemma2Wrapper(model_name_path)
    model = model.to(device)

    original_logits = model(caa_vector.unsqueeze(0))
    full_reconstruction_logits = model(full_reconstruction.unsqueeze(0))
    top_k_reconstruction_logits = model(top_k_reconstruction.unsqueeze(0))

    criterion = torch.nn.CrossEntropyLoss()
    target = torch.argmax(original_logits, dim=-1)

    original_loss = criterion(original_logits, target)
    full_reconstruction_loss = criterion(full_reconstruction_logits, target)
    top_k_reconstruction_loss = criterion(top_k_reconstruction_logits, target)

    print(f"Cross entropy loss - Original: {original_loss.item():.4f}")
    print(f"Cross entropy loss - Full reconstruction: {full_reconstruction_loss.item():.4f}")
    print(f"Cross entropy loss - Top {TOP_K_FEATURES} reconstruction: {top_k_reconstruction_loss.item():.4f}")

    original_norm = caa_vector.norm().item()
    feature_norms = {idx: feature_contributions[idx].norm().item() for idx in top_features.keys()}
    return top_features, feature_norms, original_norm

def print_top_features(behavior, layer=10, model_name_path='gemma-2b', only_combined_vector=False):
    global TOP_K_FEATURES
    top_features, feature_norms, original_norm = analyze_gemma_vector(behavior, layer, model_name_path, only_combined_vector)
    if not only_combined_vector:
        print(f"\nTop {TOP_K_FEATURES} features for '{behavior}' in layer {layer}:")
        print(f"0. Original Vector: Norm {original_norm:.4f}")
        for rank, (idx, activation) in enumerate(list(top_features.items())[:TOP_K_FEATURES], 1):
            print(f"{rank}. Feature {idx}: Activation {activation:.4f}, Norm {feature_norms[idx]:.4f}")

def calculate_loss_recovered(original, reconstruction, metric='mse'):
    if metric == 'mse':
        full_loss = torch.nn.functional.mse_loss(original, torch.zeros_like(original)).item()
        reconstruction_loss = torch.nn.functional.mse_loss(original, reconstruction).item()
        loss_recovered = (full_loss - reconstruction_loss) / full_loss
    elif metric == 'cosine':
        loss_recovered = torch.nn.functional.cosine_similarity(original.view(1, -1), reconstruction.view(1, -1)).item()
    else:
        raise ValueError("Invalid metric. Choose 'mse' or 'cosine'.")
    
    return loss_recovered * 100  # Return as percentage

if __name__ == "__main__":
    import argparse
    from behaviors import COORDINATE, ALL_BEHAVIORS

    parser = argparse.ArgumentParser(description="Analyze Gemma vector for a specific behavior.")
    parser.add_argument("--behavior", type=str, default=COORDINATE, choices=ALL_BEHAVIORS, help="Behavior to analyze")
    parser.add_argument("--layer", type=int, default=10, help="Layer to analyze")
    parser.add_argument("--model_name_path", type=str, default='gemma-2b', help="Model name path")
    parser.add_argument("--only_combined_vector", action="store_true", help="Only generate and save the combined vector")

    args = parser.parse_args()

    print_top_features(args.behavior, args.layer, args.model_name_path, args.only_combined_vector)