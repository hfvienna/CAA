"""
This script normalizes vectors for different behaviors within each layer of a model to have the same norm.
It processes vectors for all specified layers and behaviors, normalizes them, and saves the normalized vectors
to a new directory.
"""


from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os

def normalize_vectors(model_size: str, is_base: bool, n_layers: int, model_type: str):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in ALL_BEHAVIORS:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base=is_base, model_type=model_type))
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        print(f"Mean norm: {mean_norm}")
        # normalize all vectors to have the same norm
        for behavior in ALL_BEHAVIORS:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        print("Normalized norms:")
        for behavior in ALL_BEHAVIORS:
            print(f"{behavior}: {vecs[behavior].norm().item()}")
        # save the normalized vectors
        for behavior in ALL_BEHAVIORS:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize vectors for a specific model type and size.")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (e.g., '2b', '7b', '13b')")
    parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., 'gemma_1', 'llama')")
    parser.add_argument("--is_base", action="store_true", help="Whether it's a base model")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of layers in the model")

    args = parser.parse_args()

    normalize_vectors(args.model_size, args.is_base, args.n_layers, args.model_type)