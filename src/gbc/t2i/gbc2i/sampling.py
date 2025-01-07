# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from functools import partial
from PIL import Image
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import lightning.pytorch as pl
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

from ..utils import truncate_or_pad_to_length
from ..prompt import Prompt
from ..modules.text_encoders import ConcatTextEncoders
from .cfg import cfg_wrapper
from .get_sigmas import get_sigmas_from_diffusers_scheduler
from .k_diffusion import sample_euler_ancestral, DiscreteEpsDDPMDenoiser


def vae_image_postprocess(image_tensor: torch.Tensor) -> Image.Image:
    image = Image.fromarray(
        ((image_tensor * 0.5 + 0.5) * 255)
        .cpu()
        .clamp(0, 255)
        .numpy()
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    return image


@torch.no_grad()
@torch.inference_mode()
def meta_diffusion_sampling(
    unet: UNet2DConditionModel,
    te: ConcatTextEncoders,
    vae: AutoencoderKL,
    train_scheduler: EulerDiscreteScheduler,
    prompts: Prompt | list[Prompt],
    neg_prompts: Prompt | list[Prompt],
    neg_prompts_basics: Prompt | list[Prompt] | None = None,
    num_samples: int | None = None,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"] = "cycling",
    num_steps: int = 16,
    get_sigma_func: Callable[[int], list[float]] | None = None,
    cfg_wrapper_fn: Callable | None = None,
    init_noise: torch.Tensor | None = None,
    internal_sampling_func: Callable | None = None,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
    return_latent: bool = False,
):
    if seed is not None:
        pl.seed_everything(seed)

    cfg_wrapper_fn = cfg_wrapper_fn or cfg_wrapper
    internal_sampling_func = internal_sampling_func or sample_euler_ancestral

    if isinstance(prompts, Prompt):
        prompts = [prompts]
    if isinstance(neg_prompts, Prompt):
        neg_prompts = [neg_prompts]

    prompts = list(prompts)
    neg_prompts = list(neg_prompts)
    assert len(prompts) == len(neg_prompts)

    if neg_prompts_basics is not None:
        neg_prompts_basics = list(neg_prompts_basics)
    else:
        neg_prompts_basics = neg_prompts

    if num_samples is not None:
        prompts = truncate_or_pad_to_length(
            prompts, num_samples, padding_mode=padding_mode
        )
        neg_prompts = truncate_or_pad_to_length(
            neg_prompts, num_samples, padding_mode=padding_mode
        )
        neg_prompts_basics = truncate_or_pad_to_length(
            neg_prompts_basics, num_samples, padding_mode=padding_mode
        )
    else:
        num_samples = len(prompts)

    reference_param = next(unet.parameters())
    model_wrapper = DiscreteEpsDDPMDenoiser(
        unet, train_scheduler.alphas_cumprod, False, output_is_tuple=True
    ).to(reference_param)

    cfg_fn = cfg_wrapper_fn(
        prompt=prompts,
        neg_prompt=neg_prompts,
        width=width,
        height=height,
        unet=model_wrapper,
        te=te,
    )

    if get_sigma_func is None:
        sigmas = get_sigmas_from_diffusers_scheduler(
            num_steps,
            train_scheduler,
            omit_last_timestep=False,
        )
    else:
        sigmas = get_sigma_func(num_steps)
    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.tensor(sigmas)
    sigmas = sigmas.to(reference_param)

    if init_noise is None:
        init_noise = torch.randn(
            num_samples, unet.config.in_channels, height // 8, width // 8
        ).to(reference_param)
    init_x = init_noise * torch.sqrt(1 + sigmas[0] ** 2)
    # to get initial mask for sampling from gbc with no bbox
    cfg_fn_first_phase = cfg_wrapper_fn(
        prompt=prompts,
        neg_prompt=neg_prompts_basics,
        width=width,
        height=height,
        unet=model_wrapper,
        te=te,
        unet_extra_kwargs={"use_caption_mask": False},
    )
    generated_latents = internal_sampling_func(
        cfg_fn, init_x, sigmas, first_phase_model=cfg_fn_first_phase
    )
    if return_latent:
        return generated_latents

    vae_std = 1 / vae.config.scaling_factor
    vae_mean = 0.0
    generated_latents = generated_latents * vae_std + vae_mean

    image_tensors = []
    for generated_latent in generated_latents:
        image_tensors.append(vae.decode(generated_latent.unsqueeze(0)).sample)
    image_tensors = torch.concat(image_tensors)
    torch.cuda.empty_cache()
    images = []
    for image_tensor in image_tensors:
        image = vae_image_postprocess(image_tensor)
        images.append(image)
    return images


@torch.no_grad()
@torch.inference_mode()
def diffusion_sampling(
    unet: UNet2DConditionModel,
    te: ConcatTextEncoders,
    vae: AutoencoderKL,
    train_scheduler: EulerDiscreteScheduler,
    prompts: str | list[str],
    neg_prompts: str | list[str] | None = None,
    num_samples: int | None = None,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"] = "cycling",
    num_steps: int = 16,
    get_sigma_func: Callable[[int], list[float]] | None = None,
    cfg_scale: float = 3.0,
    tokenizer_padding: bool | str = "max_length",
    init_noise: torch.Tensor | None = None,
    internal_sampling_func: Callable | None = None,
    seed: int | None = None,
    width: int = 1024,
    height: int = 1024,
):
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
    cfg_wrapper_fn = partial(
        cfg_wrapper,
        cfg=cfg_scale,
        tokenizer_padding=tokenizer_padding,
    )
    return meta_diffusion_sampling(
        unet=unet,
        te=te,
        vae=vae,
        train_scheduler=train_scheduler,
        prompts=prompts,
        neg_prompts=neg_prompts,
        num_samples=num_samples,
        padding_mode=padding_mode,
        num_steps=num_steps,
        get_sigma_func=get_sigma_func,
        cfg_wrapper_fn=cfg_wrapper_fn,
        internal_sampling_func=internal_sampling_func,
        seed=seed,
        width=width,
        height=height,
    )
