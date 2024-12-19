# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import re
import importlib
import argparse
from copy import deepcopy
from omegaconf import OmegaConf
from hydra.utils import instantiate

from gbc.utils import setup_gbc_logger, save_list_to_file
from gbc.t2i import gbc_prompt_gen, load_any, GbcPrompt
from gbc.t2i.utils import truncate_or_pad_to_length


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
        default=[
            "configs/generation/t2gbc_default.yaml",
            "configs/generation/gbc2i/sampling_region_gbc_encode_with_context.yaml",
            "configs/generation/graph_transform_ex.yaml",
        ],
        help="List of config files to load.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/t2i/t2gbc_seed_with_entity_specification.yaml",
        help="List of config prompts to load. Can be either .txt or .yaml file",
    )
    parser.add_argument(
        "--neg_prompt_file",
        type=str,
        default="prompts/t2i/neg_default.yaml",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    configs = []
    for config in args.configs:
        conf = OmegaConf.load(config)
        configs.append(conf)
    config = OmegaConf.merge(*configs)

    assert (
        "image_sampling_func" in config
    ), "'image_sampling_func' must be specified in the config file"

    if (
        "model_config" in config
        and "prompt_gen_model_name_or_path" in config.model_config
    ):
        prompt_gen_model_name_or_path = (
            config.model_config.prompt_gen_model_name_or_path
        )
    else:
        prompt_gen_model_name_or_path = "graph-based-captions/GBC10M-PromptGen-200M"

    if args.save_dir is None:
        assert "save_dir" in config, (
            "'save_dir' must be specified either as a "
            "command line argument or in the config file"
        )
        args.save_dir = config.save_dir

    logger = setup_gbc_logger()

    prompt_gen_config = config.get("prompt_gen_config", {})

    # ## Prompt generation

    logger.info("Generate GBC from text prompts...")

    _, ext = os.path.splitext(args.prompt_file)

    if ext == ".txt":
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f.readlines()]
    elif ext == ".yaml":
        prompts = list(OmegaConf.load(args.prompt_file))
    else:
        raise ValueError(f"Unsupported prompt file extension: {ext}")

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

    if "graph_transform" in config:
        graph_transform = instantiate(config.graph_transform)
        gbc_graphs = [graph_transform(gbc_graph) for gbc_graph in gbc_graphs]

    # ## Image generation

    logger.info("Generate Images from GBC...")

    unet = load_any(config.model_config.unet)
    te = load_any(config.model_config.te)
    vae = load_any(config.model_config.vae)

    pos_prompts = [GbcPrompt.from_gbc_graph(gbc_graph) for gbc_graph in gbc_graphs]
    neg_prompts = read_prompt_file(args.neg_prompt_file)

    if "image_encoder" in config.model_config:
        image_encoder = load_any(config.model_config.image_encoder)
        add_kwargs = {"image_encoder": image_encoder}
    else:
        add_kwargs = {}

    sampling_func = instantiate(config.image_sampling_func)
    images = sampling_func(
        unet=unet,
        te=te,
        vae=vae,
        prompts=pos_prompts,
        neg_prompts=neg_prompts,
        **add_kwargs,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(args.save_dir, f"{i}.png"))
    logger.info(f"Saved {len(images)} images to {args.save_dir}")

    num_samples = config.image_sampling_func.get("num_samples", len(pos_prompts))
    padding_mode = config.image_sampling_func.get("padding_mode", "cycling")
    img_idxs = truncate_or_pad_to_length(
        list(range(len(gbc_graphs))), num_samples, padding_mode
    )
    gbc_graphs_generated = []
    for idx in range(num_samples):
        img_path = os.path.join(args.save_dir, f"{idx}.png")
        img_url = None
        gbc_graph = gbc_graphs[img_idxs[idx]]
        gbc_graph = deepcopy(gbc_graph).to_gbc_graph()
        gbc_graph.img_path = img_path
        gbc_graphs_generated.append(gbc_graph)
    save_list_to_file(
        gbc_graphs_generated,
        os.path.join(args.save_dir, "gbc_graphs_generated.parquet"),
    )
    logger.info("Gbc graphs with paths to generated images saved")
