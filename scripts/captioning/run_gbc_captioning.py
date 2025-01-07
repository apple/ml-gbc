# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import time
import argparse
from omegaconf import OmegaConf

from gbc.utils import (
    setup_gbc_logger,
    get_gbc_logger,
    save_list_to_file,
)
from gbc.captioning import run_gbc_captioning


def main():
    parser = argparse.ArgumentParser(description="Run model inference on an image.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/captioning/default.yaml",
        help=(
            "Path to the configuration file "
            "(default: configs/captioning/default.yaml)"
        ),
    )
    parser.add_argument(
        "--img_paths",
        type=str,
        default=["data/images/wiki/"],
        nargs="+",
        help="Paths to the input image and image folder",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="tests/outputs/captioning/gbc_wiki_images",
        required=True,
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        nargs="+",
        choices=[".json", ".jsonl", ".parquet"],
        default=[".jsonl", ".parquet"],
        help=(
            "Formats for saving the results, can be .json, .jsonl, or .parquet "
            "(default: .jsonl and .parquet)"
        ),
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Flag to indicate if input images should be saved",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=10,
        help="Frequency of saving intermediate results (default: 10)",
    )
    parser.add_argument(
        "--artifact_format",
        type=str,
        default=".jsonl",
        choices=[".json", ".jsonl", ".parquet"],
        help=(
            "Format for saving the intermediate results, can be .json, .jsonl, "
            "or .parquet (default: .jsonl)"
        ),
    )
    parser.add_argument(
        "--batch_query",
        action="store_true",
        help="Flag to indicate if query should be batched",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for batched query (default: 32)",
    )
    parser.add_argument(
        "--no_entity_query",
        action="store_true",
        help="Flag to ignore entity query",
    )
    parser.add_argument(
        "--no_composition_query",
        action="store_true",
        help="Flag to ignore composition query",
    )
    parser.add_argument(
        "--no_relation_query",
        action="store_true",
        help="Flag to ignore relation query",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use. Use default ones in config if not set.",
    )
    parser.add_argument(
        "--mask_inside_threshold",
        type=float,
        default=0.85,
        help=(
            "The threshold for determining `sub_masks` and `super_masks` "
            "at the end when the GBC graphs are created (default: 0.85)"
        ),
    )

    args = parser.parse_args()

    # Set CUDA environment variables
    if args.gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Load configuration
    config = OmegaConf.load(args.config_file)

    setup_gbc_logger()
    logger = get_gbc_logger()

    start_time = time.time()

    # Run recursive query from image files
    gbc_graphs = run_gbc_captioning(
        args.img_paths,
        captioning_cfg=config,
        include_entity_query=not args.no_entity_query,
        include_composition_query=not args.no_composition_query,
        include_relation_query=not args.no_relation_query,
        batch_query=args.batch_query,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        save_images=args.save_images,
        artifact_format=args.artifact_format,
        mask_inside_threshold=args.mask_inside_threshold,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Processing complete.")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logger.info(f"Number of graphs: {len(gbc_graphs)}")

    for save_format in args.save_format:
        save_list_to_file(
            gbc_graphs, os.path.join(args.save_dir, "gbc_graphs" + save_format)
        )


if __name__ == "__main__":
    main()
