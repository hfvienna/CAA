import torch
import matplotlib.pyplot as plt
from gemma_1_wrapper import Gemma1Wrapper
from generate_vectors import generate_save_vectors_for_behavior
from sae_lens import SAE
import os
from behaviors import get_vector_path
from behaviors import COORDINATE, ALL_BEHAVIORS, get_vector_path
from behaviors import BASE_DIR, COORDINATE, ALL_BEHAVIORS, get_vector_path
import random
import json
import numpy as np
import time

HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

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
    # Remove duplicate 'vec_layer_' if present
    if path.count('vec_layer_') > 1:
        parts = path.split('vec_layer_')
        path = os.path.join(os.path.dirname(path), f"vec_layer_{parts[-1]}")
    
    # Remove duplicate '.pt' extension if present
    if path.endswith('.pt.pt'):
        path = path[:-3]
    
    return path

def analyze_gemma_vector(behavior, layer=10, model_name_path='gemma-2b'):
    vector_path = fix_vector_path(get_vector_path(behavior, layer, model_name_path))
    normalized_dir = os.path.join(BASE_DIR, 'normalized_vectors', behavior)

    # Load the existing normalized vector
    original_path = fix_vector_path(os.path.join(normalized_dir, f"vec_layer_{layer}_{model_name_path}.pt"))
    caa_vector = torch.load(original_path)

    print(f"Analyzing vector for behavior: {behavior}")
    print(f"Vector shape: {caa_vector.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move caa_vector to device
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

    # Save top 5 feature vectors in normalized_vectors folder
    for idx in list(top_features.keys())[:5]:
        feature_vector = feature_contributions[idx]
        feature_path = fix_vector_path(os.path.join(normalized_dir, f"vec_layer_{layer}_{model_name_path}_feature_{idx}.pt"))
        torch.save(feature_vector, feature_path)

    print(f"Saved top 5 feature vectors in {normalized_dir}")
    print("To process these vectors, run:")
    print(f"python prompting_with_steering.py --layers $(seq {layer-1} {layer+1}) --multipliers -0.5 0 0.5 --type ab --model_size 2b --model_type gemma_1 --use_base_model")
    print("Then, to plot the results, run:")
    print(f"python plot_results.py --layers {layer} --multipliers -1 -0.5 0 0.5 1 --type ab --model_type gemma_1 --model_size 2b --use_base_model --behavior {behavior}")

    # Run steering analysis for the original vector and top 5 feature vectors
    print("Running steering analysis for the original vector and top 5 feature vectors...")
    for idx in ['original'] + list(top_features.keys())[:5]:
        if idx == 'original':
            vector_path = original_path
        else:
            vector_path = os.path.join(normalized_dir, f"vec_layer_{layer}_{model_name_path}_feature_{idx}.pt")
        
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

    return top_features

def print_top_10_features(behavior, layer=10, model_name_path='gemma-2b'):
    top_features = analyze_gemma_vector(behavior, layer, model_name_path)
    print(f"\nTop 10 features for '{behavior}' in layer {layer}:")
    for rank, (idx, activation) in enumerate(list(top_features.items())[:10], 1):
        print(f"{rank}. Feature {idx}: Activation {activation:.4f}")

if __name__ == "__main__":
    import sys
    from behaviors import COORDINATE, ALL_BEHAVIORS

    if len(sys.argv) > 1 and sys.argv[1] in ALL_BEHAVIORS:
        behavior = sys.argv[1]
    else:
        behavior = COORDINATE

    layer = 10 if len(sys.argv) <= 2 else int(sys.argv[2])
    model_name_path = 'gemma-2b' if len(sys.argv) <= 3 else sys.argv[3]

    print_top_10_features(behavior, layer, model_name_path)