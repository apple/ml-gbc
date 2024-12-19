# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn


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


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


class DiscreteSchedule(nn.Module):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize):
        super().__init__()
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("log_sigmas", sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = (
            dists.ge(0)
            .cumsum(dim=0)
            .argmax(dim=0)
            .clamp(max=self.log_sigmas.shape[0] - 2)
        )
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        # print(sigma)
        # print(t)
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model, alphas_cumprod, quantize, output_is_tuple: bool = False):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
        self.inner_model = model
        self.sigma_data = 1.0
        self.output_is_tuple = output_is_tuple
        self.xattn_score_cache = {}

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_out, c_in

    # This is to account for the case where inner model is diffusers unet and
    # returns tuple
    def get_eps(self, *args, store_xattn_score: bool = False, **kwargs):
        if store_xattn_score:
            self.inner_model.set_xattn_score_cache()
        result = self.inner_model(*args, **kwargs)
        if store_xattn_score:
            self.xattn_score_cache = self.inner_model.get_xattn_score_cache()
            self.inner_model.clear_xattn_score_cache()
        if self.output_is_tuple:
            return result[0]
        return result

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, sigma_cond=None, store_xattn_score=False, **kwargs):
        """
        Need sigma_cond as sigma_cond would be different from sigma
        in inversion (sigma_cond would be sigma of the next time step)
        """
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        sigma_cond = sigma_cond if sigma_cond is not None else sigma
        t = self.sigma_to_t(sigma_cond)
        eps = self.get_eps(
            input * c_in, t, store_xattn_score=store_xattn_score, **kwargs
        )
        # return eps
        return input + eps * c_out
