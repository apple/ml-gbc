# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import copy
from functools import partial
from collections.abc import Callable

import torch
import einops

from ..prompt import GbcPrompt
from ..modules.text_encoders import ConcatTextEncoders
from ..modules.attn_masks import (
    make_layer_attn_meta_dict,
    make_layer_region_mask_dict,
    LayerMaskConfig,
)
from .k_diffusion import DiscreteSchedule
from .mask_from_xattn_score import get_mask_from_xattn_score


def cfg_wrapper(
    prompt: str | list[str],
    neg_prompt: str | list[str],
    width: int,
    height: int,
    unet: DiscreteSchedule,  # should be k_diffusion wrapper
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    tokenizer_padding: bool | str = "max_length",
    unet_extra_kwargs: dict = {},
):
    # Set truncation to True for simplicity
    emb, normed_emb, pool, mask = te.encode(
        prompt + neg_prompt, padding=tokenizer_padding, truncation=True
    )
    if te.use_normed_ctx:
        emb = normed_emb
    time_ids = (
        torch.tensor([height, width, 0, 0, height, width])
        .repeat(emb.size(0), 1)
        .to(emb)
    )
    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": pool,
        }
    else:
        added_cond = None

    def cfg_fn(x, sigma, sigma_cond=None):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond, sigma_cond])
        cond, uncond = unet(
            torch.cat([x, x]),
            torch.cat([sigma, sigma]),
            sigma_cond=sigma_cond,
            encoder_hidden_states=emb,
            encoder_attention_mask=mask,
            added_cond_kwargs=added_cond,
            **unet_extra_kwargs,
        ).chunk(2)
        cfg_output = uncond + (cond - uncond) * cfg
        return cfg_output, uncond

    return cfg_fn


def region_cfg_wrapper(
    prompt: list[GbcPrompt],
    neg_prompt: list[GbcPrompt],
    width: int,
    height: int,
    unet: DiscreteSchedule,
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    n_generated_per_image: list[int] | None = None,
    image_embeds: torch.Tensor | None = None,
    reference_presence_mask: torch.BoolTensor | None = None,
    pad_to_n_bboxes: int | None = None,
    n_unet_levels: int = 3,
    tokenizer_padding: bool | str = "max_length",
    exclusive_attention: bool = False,
    unet_extra_kwargs: dict = {},
):
    assert len(prompt) == len(neg_prompt)

    gbc_prompts = prompt + neg_prompt
    prompt = [gbc_prompt.prompts for gbc_prompt in gbc_prompts]
    bboxes = [gbc_prompt.bboxes for gbc_prompt in gbc_prompts]
    adjacency = [gbc_prompts.adjacency for gbc_prompts in gbc_prompts]

    graph_attn_meta = []
    for gbc_prompt in gbc_prompts:
        # For negative gbc prompt from string we get None here
        graph_attn_meta.append(gbc_prompt.converted_labeled_adjacency)
    n_captions_per_image = torch.tensor([len(p) for p in prompt]).to(te.device)

    prompt_flattend = [item for sublist in prompt for item in sublist]

    emb, normed_emb, pool, mask = te.encode(
        prompt_flattend,
        padding=tokenizer_padding,
        truncation=True,
    )
    if te.use_normed_ctx:
        emb = normed_emb

    if n_generated_per_image is not None:
        if not isinstance(n_generated_per_image, torch.Tensor):
            n_generated_per_image = torch.tensor(n_generated_per_image)
        n_generated_per_image = torch.cat(
            [n_generated_per_image, n_generated_per_image]
        )
    else:
        n_generated_per_image = torch.ones(len(prompt), dtype=torch.long)
    n_generated_per_image = n_generated_per_image.to(te.device)

    time_ids = (
        torch.tensor([height, width, 0, 0, height, width])
        .repeat(torch.sum(n_generated_per_image).cpu().item(), 1)
        .to(emb)
    )

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": pool,
        }
    else:
        added_cond = {}

    # IP Adapter
    if image_embeds is not None:
        added_cond["image_embeds"] = [
            torch.cat([image_embeds, torch.zeros_like(image_embeds)])
        ]
        if reference_presence_mask is not None:
            # Attend to 0 as negative seems to be better than not attending
            # to anything.
            reference_presence_mask = torch.cat(
                [reference_presence_mask, reference_presence_mask]
            )

    region_mask_dict = make_layer_region_mask_dict(
        bboxes,
        adjacency,
        latent_width=width // 8,
        latent_height=height // 8,
        n_levels=n_unet_levels,
        pad_to_n_bboxes=pad_to_n_bboxes,
        skip_top_level=True,
        exclusive_attention=exclusive_attention,
        num_generated_per_image=n_generated_per_image,
    )

    def cfg_fn(
        x,
        sigma,
        sigma_cond=None,
        region_mask_dict: dict[int, torch.BoolTensor] = region_mask_dict,
        region_mask_dict_ema_to_update: dict[int, torch.FloatTensor] | None = None,
        region_mask_ema: float = 0.9,
    ):
        if sigma_cond is not None:
            sigma_cond = torch.cat([sigma_cond, sigma_cond])
        store_xattn_score = region_mask_dict_ema_to_update is not None
        cond, uncond = unet(
            torch.cat([x, x]),
            torch.cat([sigma, sigma]),
            sigma_cond=sigma_cond,
            store_xattn_score=store_xattn_score,
            encoder_hidden_states=emb,
            encoder_attention_mask=mask,
            added_cond_kwargs=added_cond,
            n_elements_per_image=n_captions_per_image,
            n_generated_per_image=n_generated_per_image,
            region_mask_dict=region_mask_dict,
            ip_region_mask_dict=region_mask_dict,
            ip_reference_presence_mask=reference_presence_mask,
            pad_to_n_elements=pad_to_n_bboxes,
            graph_attn_meta=graph_attn_meta,
            **unet_extra_kwargs,
        ).chunk(2)
        cfg_output = uncond + (cond - uncond) * cfg

        if region_mask_dict_ema_to_update is not None:
            xattn_score = unet.xattn_score_cache
            get_mask_func = partial(
                get_mask_from_xattn_score,
                graph_attn_meta=graph_attn_meta,
                n_generated_per_image=n_generated_per_image,
            )
            region_mask_dict_updated = update_region_mask_dict(
                region_mask_dict_ema_to_update,
                xattn_score,
                get_mask_func,
                region_mask_ema,
                width // 8,
                height // 8,
            )
            return cfg_output, uncond, region_mask_dict_updated

        return cfg_output, uncond

    return cfg_fn


def update_region_mask_dict(
    region_mask_dict_ema: dict[int, torch.BoolTensor],
    xattn_score: dict[int, tuple[int, torch.FloatTensor]],
    get_mask_func: Callable,
    mask_ema: float = 0.9,
    latent_width: int = 128,
    latent_height: int = 128,
):

    latent_size = latent_width * latent_height
    larger_q_length = latent_size // 4
    smaller_q_length = latent_size // 16

    region_mask_dict = {}

    for q_length, (count, sum_attn_score) in xattn_score.items():
        if q_length == larger_q_length:
            continue
        # Average across all attention maps of same shape
        avg_attn_score = sum_attn_score / count
        mask = get_mask_func(avg_attn_score)
        # Only mask from positive prompts matters
        mask_pos, mask_neg = torch.chunk(mask, 2, dim=0)
        # Shape [batch_size, query_len, key_len]
        mask = torch.cat([mask_pos, mask_pos], dim=0)
        region_mask_dict[q_length] = mask

    assert smaller_q_length in region_mask_dict
    region_mask_dict[larger_q_length] = (
        region_mask_dict[smaller_q_length]
        .unflatten(dim=1, sizes=[latent_height // 4, latent_width // 4])
        .repeat_interleave(2, dim=1)
        .repeat_interleave(2, dim=2)
        .flatten(start_dim=1, end_dim=2)
    )

    for q_length in region_mask_dict.keys():
        if q_length not in region_mask_dict_ema:
            region_mask_dict_ema[q_length] = region_mask_dict[q_length].float()
        else:
            mask = region_mask_dict_ema[q_length] * mask_ema + region_mask_dict[
                q_length
            ].float() * (1 - mask_ema)
            region_mask_dict_ema[q_length] = mask
            region_mask_dict[q_length] = mask.round().bool()
    return region_mask_dict


def prepare_prompts(
    prompt: list[GbcPrompt],
    neg_prompt: list[GbcPrompt],
    prompt_neg_prop: float,
    adj_neg_prop: float,
    all_neg_prop: float,
):
    n_repeats = 1
    aggregate_coefficients = []
    pos_neg_prompt = prompt
    if prompt_neg_prop > 0:
        n_repeats += 1
        pos_neg_prompt += neg_prompt
        aggregate_coefficients.append(prompt_neg_prop)
    if adj_neg_prop > 0:
        adj_neg_prompt = copy.deepcopy(prompt)
        for gbc_prompt in adj_neg_prompt:
            gbc_prompt.adjacency = [[]] * len(gbc_prompt.prompts)
        n_repeats += 1
        pos_neg_prompt += adj_neg_prompt
        aggregate_coefficients.append(adj_neg_prop)
    if all_neg_prop > 0:
        all_neg_prompt = copy.deepcopy(neg_prompt)
        for gbc_prompt in all_neg_prompt:
            gbc_prompt.adjacency = [[]] * len(gbc_prompt.prompts)
        n_repeats += 1
        pos_neg_prompt += all_neg_prompt
        aggregate_coefficients.append(all_neg_prop)
    return prompt, n_repeats, torch.tensor(aggregate_coefficients)


def layer_cfg_wrapper(
    prompt: list[GbcPrompt],
    neg_prompt: list[GbcPrompt],
    width: int,
    height: int,
    unet: DiscreteSchedule,
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    n_generated_per_image: list[int] | None = None,
    image_embeds: torch.Tensor | None = None,
    reference_presence_mask: torch.BoolTensor | None = None,
    layer_mask_config: LayerMaskConfig = LayerMaskConfig(),
    with_region_attention: bool = False,
    prompt_neg_prop: float = 1.0,
    adj_neg_prop: float = 0.0,
    all_neg_prop: float = 0.0,
    pad_to_n_bboxes: int | None = None,
    n_unet_levels: int = 3,
    tokenizer_padding: bool | str = "max_length",
    exclusive_region_attention: bool = False,
    unet_extra_kwargs: dict = {},
):
    assert len(prompt) == len(neg_prompt)

    gbc_prompts, n_repeats, uncond_aggregate_coefficients = prepare_prompts(
        prompt,
        neg_prompt,
        prompt_neg_prop,
        adj_neg_prop,
        all_neg_prop,
    )
    adjacency = [gbc_prompt.adjacency for gbc_prompt in gbc_prompts]
    prompt = [gbc_prompt.prompts for gbc_prompt in gbc_prompts]
    bboxes = [gbc_prompt.bboxes for gbc_prompt in gbc_prompts]

    graph_attn_meta = []
    for gbc_prompt in gbc_prompts:
        # For negative gbc prompt from string we get None here
        graph_attn_meta.append(gbc_prompt.converted_labeled_adjacency)

    if n_generated_per_image is not None:
        if not isinstance(n_generated_per_image, torch.Tensor):
            n_generated_per_image = torch.tensor(n_generated_per_image)
        n_generated_per_image = einops.repeat(
            n_generated_per_image, "b ... -> (n b) ...", n=n_repeats
        )
    else:
        n_generated_per_image = torch.tensor(
            [len(gbc_prompt) for gbc_prompt in gbc_prompts]
        )
    n_generated_per_image = n_generated_per_image.to(te.device)

    n_captions_per_image = torch.tensor([len(p) for p in prompt]).to(te.device)
    # Encode separately
    prompt_flattend = [item for sublist in prompt for item in sublist]

    emb, normed_emb, pool, mask = te.encode(
        prompt_flattend,
        padding=tokenizer_padding,
        truncation=True,
    )
    if te.use_normed_ctx:
        emb = normed_emb
    uncond_aggregate_coefficients = uncond_aggregate_coefficients.to(emb)

    time_ids = (
        torch.tensor([height, width, 0, 0, height, width])
        .repeat(torch.sum(n_generated_per_image).cpu().item(), 1)
        .to(emb)
    )

    # sdxl
    if pool is not None:
        added_cond = {
            "time_ids": time_ids,
            "text_embeds": pool,
        }
    else:
        added_cond = {}
    if image_embeds is not None:
        added_cond["image_embeds"] = [
            torch.cat(
                [
                    image_embeds,
                    einops.repeat(
                        torch.zeros_like(image_embeds),
                        "b ... -> (n b) ...",
                        n=n_repeats - 1,
                    ),
                ]
            )
        ]
        if reference_presence_mask is not None:
            reference_presence_mask = torch.cat(
                [
                    reference_presence_mask,
                    einops.repeat(
                        reference_presence_mask, "b ... -> (n b) ...", n=n_repeats - 1
                    ),
                ]
            )

    layer_attn_meta_dict = make_layer_attn_meta_dict(
        bboxes,
        adjacency,
        latent_width=width // 8,
        latent_height=height // 8,
        layer_mask_config=layer_mask_config,
        n_levels=n_unet_levels,
        pad_to_n_bboxes=pad_to_n_bboxes,
        skip_top_level=True,
        num_generated_per_image=n_generated_per_image,
    )

    if with_region_attention:
        region_mask_dict = make_layer_region_mask_dict(
            bboxes,
            adjacency,
            latent_width=width // 8,
            latent_height=height // 8,
            n_levels=n_unet_levels,
            pad_to_n_bboxes=pad_to_n_bboxes,
            skip_top_level=True,
            num_generated_per_image=n_generated_per_image,
            exclusive_attention=exclusive_region_attention,
        )
    else:
        region_mask_dict = None

    def cfg_fn(x, sigma, sigma_cond=None):
        # print(x.shape)
        if sigma_cond is not None:
            sigma_cond = einops.repeat(sigma_cond, "b ... -> (n b) ...", n=n_repeats)
        x = einops.repeat(x, "b ... -> (n b) ...", n=n_repeats)
        sigma = einops.repeat(sigma, "b ... -> (n b) ...", n=n_repeats)
        denoised = unet(
            x,
            sigma,
            sigma_cond=sigma_cond,
            encoder_hidden_states=emb,
            encoder_attention_mask=mask,
            added_cond_kwargs=added_cond,
            n_elements_per_image=n_captions_per_image,
            n_generated_per_image=n_generated_per_image,
            region_mask_dict=region_mask_dict,
            ip_region_mask_dict=region_mask_dict,
            ip_reference_presence_mask=reference_presence_mask,
            layer_attn_meta_dict=layer_attn_meta_dict,
            pad_to_n_elements=pad_to_n_bboxes,
            graph_attn_meta=graph_attn_meta,
            **unet_extra_kwargs,
        )
        assert n_repeats > 1
        cond_unconds = einops.rearrange(denoised, "(n b) ... -> n b ...", n=n_repeats)
        cond = cond_unconds[0]
        uncond = torch.tensordot(
            uncond_aggregate_coefficients, cond_unconds[1:], dims=1
        )
        cfg_output = uncond + (cond - uncond) * cfg
        return cfg_output, uncond

    return cfg_fn
