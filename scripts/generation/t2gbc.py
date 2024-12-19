# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import re
import importlib
import argparse
from omegaconf import OmegaConf

from gbc.utils import setup_gbc_logger, save_list_to_file
from gbc.t2i import gbc_prompt_gen


def read_prompt_file(prompt_file):
    _, ext = os.path.splitext(prompt_file)
    if ext == ".txt":
        with open(prompt_file) as f:
            prompts = [line.strip() for line in f.readlines()]
    elif ext == ".yaml":
        prompts = list(OmegaConf.load(prompt_file))
    else:
        raise ValueError(f"Unsupported prompt file extension: {ext}")
    return prompts


def extract_nested_config(args, config, section_name, key_names=[]):
    """Extract and overwrite a specific section of the config."""
    section_config = config.get(section_name, {})
    for key, value in vars(args).items():
        if value is not None and (key in section_config or key in key_names):
            section_config[key] = value
    return section_config


def postprocess_prompts(prompts: list[str]) -> tuple[list[str], list[list[str]]]:
    """
    Post-process prompts by removing brackets and extracting bracketed words.

    Parameters
    ----------
    prompts
        List of input prompts.

    Returns
    -------
    tuple[list[str], list[list[str]]]
        - Modified prompts without brackets.
        - List of words found within brackets for each prompt.
    """
    modified_prompts = []
    bracketed_words = []

    for prompt in prompts:
        # Extract words inside brackets
        words_in_brackets = re.findall(r"\[(.*?)\]", prompt)
        bracketed_words.append(words_in_brackets)

        # Remove brackets from the prompt
        cleaned_prompt = re.sub(r"\[.*?\]", lambda match: match.group(0)[1:-1], prompt)
        modified_prompts.append(cleaned_prompt)

    return modified_prompts, bracketed_words


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["configs/generation/t2gbc_default.yaml"],
        help="List of config files to load.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/t2i/t2gbc_seed_with_entity_specification.yaml",
        help="List of config prompts to load. Can be either .txt or .yaml file",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="Path to the save file. Overrides the config if specified.",
    )
    # Arguments for model_config
    parser.add_argument(
        "--prompt_gen_model_name_or_path",
        type=str,
        default=None,
        help="Path to the prompt generation model.",
    )
    # Arguments for prompt_gen_config
    parser.add_argument(
        "--allow_composition", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--star_graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    args = parser.parse_args()

    configs = []
    for config in args.configs:
        conf = OmegaConf.load(config)
        configs.append(conf)
    config = OmegaConf.merge(*configs)

    if args.prompt_gen_model_name_or_path is not None:
        prompt_gen_model_name_or_path = args.prompt_gen_model_name_or_path
    elif (
        "model_config" in config
        and "prompt_gen_model_name_or_path" in config.model_config
    ):
        prompt_gen_model_name_or_path = (
            config.model_config.prompt_gen_model_name_or_path
        )
    else:
        prompt_gen_model_name_or_path = "graph-based-captions/GBC10M-PromptGen-200M"

    if args.save_file is None:
        assert "save_file" in config, (
            "'save_file' must be specified either as a "
            "command line argument or in the config file"
        )
        args.save_file = config.save_file

    # Overwrite prompt_gen_config with values from args
    prompt_gen_config_names = [
        "allow_composition",
        "star_graph",
        "verbose",
        "seed",
        "num_samples",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
        "torch_dtype",
        "device",
        "attn_implementation",
    ]
    prompt_gen_config = extract_nested_config(
        args, config, "prompt_gen_config", key_names=prompt_gen_config_names
    )

    logger = setup_gbc_logger()

    if prompt_gen_config.get("verbose", False):
        logger.info(f"Running with configuration:\n {prompt_gen_config}")

    prompts = read_prompt_file(args.prompt_file)
    prompts, entities_to_describe = postprocess_prompts(prompts)

    if (
        "torch_dtype" in config.prompt_gen_config
        and config.prompt_gen_config.torch_dtype.startswith("torch")
    ):
        torch = importlib.import_module("torch")
        torch_dtype = eval(
            config.prompt_gen_config.pop("torch_dtype"), {"torch": torch}
        )

    gbc_graphs = gbc_prompt_gen(
        prompt_gen_model_name_or_path,
        prompts,
        entity_lists=entities_to_describe,
        torch_dtype=torch_dtype,
        **config.prompt_gen_config,
    )

    save_dir = os.path.dirname(args.save_file)
    if save_dir.strip() and save_dir.strip() != ".":
        os.makedirs(save_dir, exist_ok=True)
    save_list_to_file(gbc_graphs, args.save_file)
