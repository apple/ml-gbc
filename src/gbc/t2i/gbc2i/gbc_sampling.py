# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from functools import partial
from typing import Literal
from collections.abc import Callable
from omegaconf import DictConfig
from PIL import Image

import torch
import lightning.pytorch as pl
from transformers import CLIPVisionModelWithProjection
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

from ..utils import truncate_or_pad_to_length
from ..prompt import GbcPrompt
from ..modules.text_encoders import ConcatTextEncoders
from ..modules.attn_masks import LayerMaskConfig
from .cfg import region_cfg_wrapper, layer_cfg_wrapper
from .sampling import meta_diffusion_sampling
from .k_diffusion import sample_euler_ancestral


@torch.no_grad()
@torch.inference_mode()
def gbc_diffusion_sampling(
    unet: UNet2DConditionModel,
    te: ConcatTextEncoders,
    vae: AutoencoderKL,
    train_scheduler: EulerDiscreteScheduler,
    prompts: list[dict | DictConfig | GbcPrompt],
    neg_prompts: list[str | dict | DictConfig | GbcPrompt] | None = None,
    pad_to_n_bboxes: int | None = None,  # for padding
    cfg_scale: float = 3.0,
    tokenizer_padding: bool | str = "max_length",
    n_generated_per_image: list[int] | None = None,
    use_region_attention: bool = True,
    exclusive_region_attention: bool = False,
    labels_in_neg: bool = False,
    use_layer_attention: bool = True,
    layer_mask_config: LayerMaskConfig = LayerMaskConfig(),
    return_flattened: bool = False,
    return_with_bbox: bool = False,
    prompt_neg_prop: float = 1.0,
    adj_neg_prop: float = 0.0,
    all_neg_prop: float = 0.0,
    concat_ancestor_prompts: bool = False,
    image_encoder: CLIPVisionModelWithProjection | None = None,
    # Arguments for meta_diffusion_sampling
    num_samples: int | None = None,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"] = "cycling",
    num_steps: int = 16,
    get_sigma_func: Callable[[int], list[float]] | None = None,
    internal_sampling_func: Callable | None = None,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
) -> list[Image.Image] | list[list[Image.Image]]:

    # seed here as we may initialize initial latent
    if seed is not None:
        pl.seed_everything(seed)

    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(neg_prompts, str):
        neg_prompts = [neg_prompts]
    if neg_prompts is None:
        neg_prompts = [""]
    if len(prompts) != len(neg_prompts):
        assert len(prompts) % len(neg_prompts) == 0
        n_neg_repeats = len(prompts) // len(neg_prompts)
        neg_prompts = list(neg_prompts) * n_neg_repeats

    reference_param = next(unet.parameters())
    gbc_prompts = []
    neg_gbc_prompts = []
    neg_gbc_prompts_basics = []

    # Use the first tokenizer for edge conversion
    tokenizer = te.tokenizers[0]

    for gbc_prompt, neg_gbc_prompt in zip(prompts, neg_prompts):
        if not isinstance(gbc_prompt, GbcPrompt):
            if isinstance(gbc_prompt, str):
                gbc_prompt = GbcPrompt.from_string(gbc_prompt)
            else:
                gbc_prompt = GbcPrompt(**gbc_prompt)
        if concat_ancestor_prompts:
            gbc_prompt = gbc_prompt.concat_ancestor_prompts(
                mask_out_concat=True, tokenizer=te.tokenizers[0]
            )
        gbc_prompt = gbc_prompt.sort_with_topological_order()
        gbc_prompt.convert_bboxes(reference_param)
        gbc_prompt.convert_adjacency(tokenizer)
        if image_encoder is not None:
            gbc_prompt.prepare_image_embeds(image_encoder)
        gbc_prompts.append(gbc_prompt)
        if isinstance(neg_gbc_prompt, str) or (
            isinstance(neg_gbc_prompt, list) and isinstance(neg_gbc_prompt[0], str)
        ):
            neg_gbc_prompt_basics = gbc_prompt.new_gbc_prompt_from_str(
                neg_gbc_prompt, neg_prompt=False
            )
            neg_gbc_prompt = gbc_prompt.new_gbc_prompt_from_str(
                neg_gbc_prompt, neg_prompt=labels_in_neg
            )
        elif not isinstance(neg_gbc_prompt, GbcPrompt):
            neg_gbc_prompt = GbcPrompt(**neg_gbc_prompt)
            neg_gbc_prompt_basics = neg_gbc_prompt

        neg_gbc_prompt = neg_gbc_prompt.sort_with_topological_order()
        neg_gbc_prompt.convert_bboxes(reference_param)
        neg_gbc_prompts.append(neg_gbc_prompt)

        neg_gbc_prompt_basics = neg_gbc_prompt_basics.sort_with_topological_order()
        neg_gbc_prompt_basics.convert_bboxes(reference_param)
        neg_gbc_prompts_basics.append(neg_gbc_prompt_basics)

    n_unet_levels = len(unet.down_blocks)
    if n_generated_per_image is not None:
        n_generated_per_image = list(n_generated_per_image)

    # We get the right set of prompts directly here
    if num_samples is not None:
        gbc_prompts = truncate_or_pad_to_length(
            gbc_prompts, num_samples, padding_mode=padding_mode
        )
        neg_gbc_prompts = truncate_or_pad_to_length(
            neg_gbc_prompts, num_samples, padding_mode=padding_mode
        )
        neg_gbc_prompt_basics = truncate_or_pad_to_length(
            neg_gbc_prompts_basics, num_samples, padding_mode=padding_mode
        )
        if n_generated_per_image is not None:
            n_generated_per_image = truncate_or_pad_to_length(
                n_generated_per_image, num_samples, padding_mode=padding_mode
            )

    image_embeds = []
    reference_presence_mask = []
    has_image_embeds = False
    for gbc_prompt in gbc_prompts:
        if gbc_prompt.ref_image_embeds is not None:
            image_embeds.append(gbc_prompt.ref_image_embeds)
            reference_presence_mask.append(gbc_prompt.ref_image_idx_mask)
            has_image_embeds = True
        else:
            image_embeds.append(None)
            reference_presence_mask.append(None)
    if has_image_embeds:
        image_embed_shape = None
        for image_embed in image_embeds:
            if image_embed is not None:
                image_embed_shape = image_embed.shape[1:]
                break
        for i, (image_embed, gbc_prompt) in enumerate(zip(image_embeds, gbc_prompts)):
            if image_embed is None:
                image_embeds[i] = torch.zeros(
                    len(gbc_prompt.prompts),
                    *image_embed_shape,
                    dtype=reference_param.dtype,
                    device=reference_param.device,
                )
                reference_presence_mask[i] = torch.zeros(
                    len(gbc_prompt.prompts), dtype=bool, device=reference_param.device
                )
        # Unsqueeze for image dimension for each generated image
        image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(1)
        reference_presence_mask = torch.cat(reference_presence_mask, dim=0)
    if not has_image_embeds:
        image_embeds = None
        reference_presence_mask = None

    if use_layer_attention:
        if not isinstance(layer_mask_config, LayerMaskConfig):
            layer_mask_config = LayerMaskConfig(**layer_mask_config)
        cfg_wrapper_fn = partial(
            layer_cfg_wrapper,
            cfg=cfg_scale,
            prompt_neg_prop=prompt_neg_prop,
            adj_neg_prop=adj_neg_prop,
            all_neg_prop=all_neg_prop,
            tokenizer_padding=tokenizer_padding,
            n_generated_per_image=n_generated_per_image,
            pad_to_n_bboxes=pad_to_n_bboxes,
            n_unet_levels=n_unet_levels,
            layer_mask_config=layer_mask_config,
            with_region_attention=use_region_attention,
            exclusive_region_attention=exclusive_region_attention,
            image_embeds=image_embeds,
            reference_presence_mask=reference_presence_mask,
        )
        if n_generated_per_image is not None:
            num_layers = sum(n_generated_per_image)
        else:
            num_layers = sum([len(p) for p in gbc_prompts])
        init_noise = torch.randn(
            num_layers, unet.config.in_channels, height // 8, width // 8
        ).to(reference_param)
    elif use_region_attention:
        cfg_wrapper_fn = partial(
            region_cfg_wrapper,
            cfg=cfg_scale,
            tokenizer_padding=tokenizer_padding,
            n_generated_per_image=n_generated_per_image,
            pad_to_n_bboxes=pad_to_n_bboxes,
            n_unet_levels=n_unet_levels,
            exclusive_attention=exclusive_region_attention,
            image_embeds=image_embeds,
            reference_presence_mask=reference_presence_mask,
        )
        if n_generated_per_image is not None:
            num_layers = sum(n_generated_per_image)
            init_noise = torch.randn(
                num_layers, unet.config.in_channels, height // 8, width // 8
            ).to(reference_param)
        else:
            init_noise = None

    internal_sampling_func = internal_sampling_func or sample_euler_ancestral

    images = meta_diffusion_sampling(
        unet,
        te,
        vae,
        train_scheduler,
        gbc_prompts,
        neg_gbc_prompts,
        neg_prompts_basics=neg_gbc_prompts_basics,
        cfg_wrapper_fn=cfg_wrapper_fn,
        init_noise=init_noise,
        internal_sampling_func=internal_sampling_func,
        num_samples=num_samples,
        padding_mode=padding_mode,
        num_steps=num_steps,
        get_sigma_func=get_sigma_func,
        width=width,
        height=height,
    )

    if n_generated_per_image is None:
        if use_layer_attention:
            n_generated_per_image = [
                len(gbc_prompt.prompts) for gbc_prompt in gbc_prompts
            ]
        else:
            n_generated_per_image = [1] * len(gbc_prompts)

    images_with_bbox = []
    if return_with_bbox:
        img_idx = 0
        for n_generated, gbc_prompt in zip(n_generated_per_image, gbc_prompts):
            images_with_bbox.append(gbc_prompt.add_bbox_to_image(images[img_idx]))
            img_idx += n_generated

    if not return_flattened:
        # regroup images
        start = 0
        stacked_images = []
        for num_generated in n_generated_per_image:
            stacked_images.append(images[start : start + num_generated])
            start += num_generated
        images = stacked_images

    images = images + images_with_bbox

    return images
