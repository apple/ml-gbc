# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from tqdm import trange

import torch


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but "
            f"target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
@torch.inference_mode()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    use_extra_args_step: list[int] | None = None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    image_to_noise: bool = False,
    # For sampling from GBC without bounding boxes
    get_mask_from_xattn_scores: bool = False,
    n_first_phase_steps: int = 12,
    first_phase_start_compute_ema_step: int = 6,
    first_phase_model=None,
    region_mask_ema: float = 0.9,
):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    # computed if get mask from xattn scores
    region_mask_dict = None
    region_mask_dict_ema = {}

    # To store cross attention map
    if get_mask_from_xattn_scores and n_first_phase_steps > 0:
        if first_phase_model is None:
            first_phase_model = model
        x_clone = x.clone()
        for i in trange(n_first_phase_steps):
            sigma_cond = sigmas[i + 1] if image_to_noise else sigmas[i]
            if use_extra_args_step is not None and i not in use_extra_args_step:
                extra_args_to_use = {}
            else:
                extra_args_to_use = extra_args
            if i >= first_phase_start_compute_ema_step:
                denoised, _, region_mask_dict = first_phase_model(
                    x,
                    sigmas[i] * s_in,
                    sigma_cond=sigma_cond * s_in,
                    # updated in place
                    region_mask_dict_ema_to_update=region_mask_dict_ema,
                    region_mask_ema=region_mask_ema,
                    **extra_args_to_use,
                )
            else:
                denoised, _ = first_phase_model(
                    x,
                    sigmas[i] * s_in,
                    sigma_cond=sigma_cond * s_in,
                    **extra_args_to_use,
                )
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        x = x_clone

    # The main sampling loop
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cond = sigmas[i + 1] if image_to_noise else sigmas[i]
        if use_extra_args_step is not None and i not in use_extra_args_step:
            extra_args_to_use = {}
        else:
            extra_args_to_use = extra_args
        if region_mask_dict:
            extra_args_to_use["region_mask_dict"] = region_mask_dict
        if get_mask_from_xattn_scores and i >= n_first_phase_steps:
            denoised, _, region_mask_dict = first_phase_model(
                x,
                sigmas[i] * s_in,
                sigma_cond=sigma_cond * s_in,
                region_mask_dict_ema_to_update=region_mask_dict_ema,  # updated in place
                region_mask_ema=region_mask_ema,
                **extra_args_to_use,
            )
        else:
            denoised, _ = model(
                x,
                sigmas[i] * s_in,
                sigma_cond=sigma_cond * s_in,
                **extra_args_to_use,
            )
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
