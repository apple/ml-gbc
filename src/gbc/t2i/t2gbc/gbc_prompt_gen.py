# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from functools import cache
from typing import Literal

import torch
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import truncate_or_pad_to_length
from .generation import generate_text
from .sampling_constraint import create_prefix_allowed_tokens_fn
from .graph_parse import parse_gbc_graph


@cache
def load_gbc_prompt_gen(
    pretrained_model_name_or_path: str,
    torch_dtype: str | torch.dtype = torch.float32,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = (
        AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        .eval()
        .to(device)
    )
    return tokenizer, model


@torch.no_grad()
@torch.inference_mode()
def gbc_prompt_gen(
    pretrained_model_name_or_path: str,
    prompts: list[str],
    num_samples: int | None = None,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"] = "cycling",
    allow_composition: bool = True,
    star_graph: bool = False,
    entity_lists: list[list[str]] | None = None,
    verbose: bool = False,
    seed: int | None = None,
    # Generation config
    temperature: float = 1,
    top_p: float = 0.95,
    top_k: int = 60,
    repetition_penalty: float = 1,
    max_new_tokens: int = 4096,
    generation_kwargs: dict = {},
    # For model loading
    torch_dtype: str | torch.dtype = torch.float32,
    device: str = "cpu",
    attn_implementation: str = "sdpa",
):

    if seed is not None:
        pl.seed_everything(seed)

    if num_samples is not None:
        prompts = truncate_or_pad_to_length(
            prompts, num_samples, padding_mode=padding_mode
        )
        if entity_lists is not None:
            entity_lists = truncate_or_pad_to_length(
                entity_lists, num_samples, padding_mode=padding_mode
            )

    tokenizer, model = load_gbc_prompt_gen(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        device=device,
        attn_implementation=attn_implementation,
    )

    data = []

    for idx, prompt in enumerate(prompts):

        input = f"""Node #0 image
        type: image
        is_leave: False
        desc: {prompt}
        parents:
        bbox: left: 0.0000000, top: 0.0000000, right: 1.0000000, bottom: 1.0000000
        """
        input = "\n".join(line.strip() for line in input.splitlines())

        prev = input

        entity_lists_i = [entity_lists[idx]] if entity_lists is not None else []

        prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(
            tokenizer,
            allow_composition=allow_composition,
            star_graph=star_graph,
            entity_lists=entity_lists_i,
        )
        model_kwargs = {
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        }

        result = generate_text(
            model,
            tokenizer,
            input,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            stream_output=verbose,
            model_kwargs=model_kwargs,
            generation_kwargs=generation_kwargs,
        )

        if verbose:
            print()
        prev = ""
        for k in result:
            if len(k) > len(prev):
                if verbose:
                    print(k[len(prev) :], end="", flush=True)
                prev = k
        result = prev
        if verbose:
            print()
        gbc_graph = parse_gbc_graph(result, prompt, verbose=verbose)
        data.append(gbc_graph)

    return data
