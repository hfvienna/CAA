"""
Plot results for SAE vectors

Usage:
python sae_plot_results.py --layers 14 --multipliers -1 0 1 --type ab --model_type gemma_2 --model_size 2b --use_base_model
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from typing import List
from steering_settings import SteeringSettings
from behaviors import ALL_BEHAVIORS, get_results_dir, get_analysis_dir, HUMAN_NAMES
from utils.helpers import set_plotting_settings

set_plotting_settings()

def load_results(file_path: str) -> List[dict]:
    with open(file_path, "r") as f:
        return json.load(f)

def get_avg_key_prob(results: List[dict], key: str) -> float:
    if not results:
        return 0.0
    return sum(1 if result[key] == '(A)' else 0 for result in results) / len(results)

def get_data(layer: int, multiplier: int, settings: SteeringSettings) -> List[dict]:
    directory = get_results_dir(settings.behavior)
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for filter {settings}", filenames)
    if len(filenames) == 0:
        print(f"[WARN] no filenames found for filter {settings}")
        return []
    with open(filenames[0], "r") as f:
        return json.load(f)

def plot_ab_results_for_layer(layer: int, multipliers: List[float], settings: SteeringSettings):
    save_to = os.path.join(
        "sae_analysis",
        f"layer={layer}_behavior={settings.behavior}_multipliers={'_'.join(map(str, multipliers))}_model_type={settings.model_type}_model_size={settings.model_size}_use_base_model={settings.use_base_model}.png",
    )
    print(f"Saving to: {save_to}")  # Debug print
    plt.clf()
    plt.figure(figsize=(10, 3.5))
    all_results = {}
    for multiplier in multipliers:
        results = get_data(layer, multiplier, settings)
        avg_key_prob = get_avg_key_prob(results, "a_prob")
        plt.plot(
            [multiplier],
            [avg_key_prob],
            marker="o",
            linestyle="solid",
            markersize=10,
            linewidth=3,
            label=f"Multiplier {multiplier}"  # Add a label for each plot element
        )
        all_results[multiplier] = avg_key_prob
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Multiplier")
    plt.ylabel("p(answer matching behavior)")
    plt.xticks(ticks=multipliers, labels=multipliers)
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"{HUMAN_NAMES[settings.behavior]} - {settings.get_formatted_model_name()}", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    plt.savefig(save_to, format="png")
    plt.close()
    print(f"Saved plot to {save_to}")

def plot_effect_on_behaviors(layer: int, multipliers: List[float], behaviors: List[str], settings: SteeringSettings):
    plt.clf()
    plt.figure(figsize=(6, 3))
    multiplier_range = f"{min(multipliers)}to{max(multipliers)}"
    save_to = os.path.join(
        "sae_analysis",
        f"layer={layer}_behaviors=multiple_type={settings.type}_multipliers={multiplier_range}_model_type={settings.model_type}_model_size={settings.model_size}_use_base_model={settings.use_base_model}.png",
    )
    all_results = []
    for behavior in behaviors:
        results = []
        for mult in multipliers:
            settings.behavior = behavior
            data = get_data(layer, mult, settings)
            print(f"Behavior: {behavior}, Multiplier: {mult}, Data: {data}")
            avg_key_prob = get_avg_key_prob(data, "a_prob")
            print(f"Avg key prob: {avg_key_prob}")
            results.append(avg_key_prob * 100)
        all_results.append(results)
        print(f"Results for {behavior}: {results}")
        plt.plot(multipliers, results, marker='o', label=HUMAN_NAMES[behavior])
    
    plt.legend()
    plt.xlabel("Multiplier")
    plt.ylabel("p(answer matching behavior) %")
    plt.title(f"Layer {layer} - {settings.get_formatted_model_name()}", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    plt.savefig(save_to, format="png")
    plt.close()
    print(f"Saved plot to {save_to}")

    # Save data as txt
    with open(save_to.replace(".png", ".txt"), "w") as f:
        f.write("Behavior\t" + "\t".join(map(str, multipliers)) + "\n")
        for behavior, results in zip(behaviors, all_results):
            f.write(f"{behavior}\t" + "\t".join(map(str, results)) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--behaviors", type=str, nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument("--type", type=str, required=True, choices=["ab"])
    parser.add_argument("--model_type", type=str, choices=["llama", "gemma_1", "gemma_2"], default="llama")
    parser.add_argument("--model_size", type=str, choices=["7b", "13b", "2b"], default="7b")
    parser.add_argument("--use_base_model", action="store_true", default=False)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.model_type = args.model_type
    steering_settings.model_size = args.model_size
    steering_settings.use_base_model = args.use_base_model

    for layer in args.layers:
        plot_effect_on_behaviors(layer, args.multipliers, args.behaviors, steering_settings)