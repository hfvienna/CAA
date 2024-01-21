"""
Script to plot PCA of constrastive activations

Usage:
python plot_activations.py --behavior sycophancy --layer 10 --use_base_model --model_size 7b
"""

import json
import torch as t
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.decomposition import PCA
from behaviors import get_activations_path, get_ab_data_path, get_analysis_dir
from utils.helpers import get_model_path

DATASET_FILE = os.path.join("preprocessed_data", "generate_dataset.json")


def save_activation_projection_pca(behavior: str, layer: int, model_name_path: str):
    title = f"PCA of contrastive activations for {behavior} at layer {layer}"
    fname = f"pca_{behavior}_layer_{layer}.pdf"
    save_dir = get_analysis_dir(behavior)

    # Loading activations
    activations_pos = t.load(
        get_activations_path(behavior, layer, model_name_path, "pos")
    )
    activations_neg = t.load(
        get_activations_path(behavior, layer, model_name_path, "neg")
    )

    # Getting letters
    with open(get_ab_data_path(behavior), "r") as f:
        data = json.load(f)

    letters_pos = [item["answer_matching_behavior"][1] for item in data]
    letters_neg = [item["answer_not_matching_behavior"][1] for item in data]

    plt.clf()
    activations = t.cat([activations_pos, activations_neg], dim=0)
    activations_np = activations.cpu().numpy()

    # PCA projection
    pca = PCA(n_components=2)
    projected_activations = pca.fit_transform(activations_np)

    # Splitting back into activations1 and activations2
    activations_pos_projected = projected_activations[: activations_pos.shape[0]]
    activations_neg_projected = projected_activations[activations_pos.shape[0] :]

    # Visualization
    for i, (x, y) in enumerate(activations_pos_projected):
        if letters_pos[i] == "A":
            plt.scatter(x, y, color="blue", marker="o", alpha=0.4)
        elif letters_pos[i] == "B":
            plt.scatter(x, y, color="blue", marker="x", alpha=0.4)

    for i, (x, y) in enumerate(activations_neg_projected):
        if letters_neg[i] == "A":
            plt.scatter(x, y, color="red", marker="o", alpha=0.4)
        elif letters_neg[i] == "B":
            plt.scatter(x, y, color="red", marker="x", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {behavior} - A",
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="blue",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {behavior} - B",
    )
    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {behavior} - A",
    )
    scatter4 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="red",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {behavior} - B",
    )

    plt.legend(handles=[scatter1, scatter2, scatter3, scatter4])
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.savefig(os.path.join(save_dir, fname), format="pdf")

    # Print ratio between first and second principal component values
    print(f"Ratio between first and second principal component values: {pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    args = parser.parse_args()
    model_name_path = get_model_path(args.model_size, args.use_base_model)
    args = parser.parse_args()
    save_activation_projection_pca(
        args.behavior,
        args.layer,
        model_name_path,
    )
