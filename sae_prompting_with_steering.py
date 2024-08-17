"""
Use SAE vectors to steer the model

Usage:
python sae_prompting_with_steering.py --behaviors sycophancy --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --use_base_model --model_size 7b
"""

import json
import torch
from llama_wrapper import LlamaWrapper
from gemma_1_wrapper import Gemma1Wrapper
from gemma_2_wrapper import Gemma2Wrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_ab_test_data,
    get_system_prompt,
    ALL_BEHAVIORS,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False, overwrite_vectors=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = os.path.join("sae_vector_results", settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
    }
    model = Gemma2Wrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        vector_path = os.path.join("sae_vector_normalized", f"normalized_{settings.behavior}.pt")
        print(f"Loading vector from: {vector_path}")
        vector = torch.load(vector_path)
        if settings.model_size != "7b":
            vector = vector.half()
        vector = vector.to(model.device)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            print(f"Saving results to: {save_filename}")
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b", "2b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--model_type", type=str, choices=["llama", "gemma_1", "gemma_2"], default="llama")
    parser.add_argument("--overwrite_vectors", action="store_true", default=False, help="Overwrite existing vectors if they exist")
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.model_type = args.model_type
    steering_settings.overwrite_vectors = args.overwrite_vectors

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
            overwrite_vectors=args.overwrite_vectors
        )