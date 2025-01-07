# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import os
from copy import deepcopy
from omegaconf import OmegaConf
from hydra.utils import instantiate

from gbc.utils import setup_gbc_logger, load_list_from_file, save_list_to_file
from gbc.data import GbcGraphFull
from gbc.t2i import load_any, GbcPrompt
from gbc.t2i.utils import truncate_or_pad_to_length


def read_prompt_file(prompt_file, config):
    if prompt_file.endswith(".yaml"):
        prompts = OmegaConf.load(prompt_file)
        return prompts, [None] * len(prompts)
    gbc_graphs = load_list_from_file(prompt_file, GbcGraphFull)
    if "graph_transform" in config:
        graph_transform = instantiate(config.graph_transform)
        gbc_graphs = [graph_transform(gbc_graph) for gbc_graph in gbc_graphs]
    prompts = [GbcPrompt.from_gbc_graph(gbc_graph) for gbc_graph in gbc_graphs]
    return prompts, gbc_graphs


def read_prompt_files(prompt_files, config):
    all_prompts = []
    gbc_graphs = []
    for prompt_file in prompt_files:
        prompts, graphs = read_prompt_file(prompt_file, config)
        all_prompts.extend(prompts)
        gbc_graphs.extend(graphs)
    return all_prompts, gbc_graphs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["configs/generation/gbc2i/sampling_base.yaml"],
    )
    parser.add_argument(
        "--prompt_files",
        type=str,
        nargs="+",
        default=["prompts/t2i/t2gbc_seed.yaml"],
    )
    parser.add_argument(
        "--neg_prompt_files",
        type=str,
        nargs="+",
        default=["prompts/t2i/neg_default.yaml"],
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

    if args.save_dir is None:
        assert "save_dir" in config, (
            "'save_dir' must be specified either as a "
            "command line argument or in the config file"
        )
        args.save_dir = config.save_dir

    pos_prompts, gbc_graphs = read_prompt_files(args.prompt_files, config)
    neg_prompts, _ = read_prompt_files(args.neg_prompt_files, config)

    logger = setup_gbc_logger()
    unet = load_any(config.model_config.unet)
    te = load_any(config.model_config.te)
    vae = load_any(config.model_config.vae)

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
        if gbc_graph is not None:
            gbc_graph = deepcopy(gbc_graph).to_gbc_graph()
            gbc_graph.img_path = img_path
            gbc_graphs_generated.append(gbc_graph)
    if gbc_graphs_generated:
        save_list_to_file(
            gbc_graphs_generated,
            os.path.join(args.save_dir, "gbc_graphs_generated.parquet"),
        )
        logger.info("Gbc graphs with paths to generated images saved")
