# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from lightning.pytorch import LightningModule

from gbc.t2i.utils import remove_none, load_any


class BaseTextEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.text_model = None

    def tokenize(self, text: str) -> list[int] | list[list[int]] | torch.LongTensor:
        raise NotImplementedError

    def encode(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tokenizer_outputs: list[dict[str, torch.Tensor]]):
        raise NotImplementedError


@dataclass
class TextModelExtraConfig:
    concat_bucket: int = 0
    use_pooled: bool = False
    layer_idx: int = -1
    need_mask: bool = False
    disable_autocast: bool = False


class ConcatTextEncoders(BaseTextEncoder):
    def __init__(
        self,
        tokenizers: list[str] = [],
        text_model_and_configs: list[
            tuple[nn.Module | dict, TextModelExtraConfig | dict]
        ] = [],
        zero_for_padding: bool = True,
        max_length: int = 256,
        use_normed_ctx: bool = False,
    ):
        """
        A text encoder wrapper for multiple tokenizers and text models.
        Can support tricky concat config like what SD3 need

        SDXL:
            tes: [CLIP-L, openCLIP-G]
            concat_bucket: [0, 0]
            use_pooled: [True, True]
            layer_idx: [-1, -2]
        SD3:
            tes: [CLIP-L, openCLIP-G, T5-xxl]
            concat_bucket: [0, 0, 1]
            use_pooled: [True, True, False]
        """
        super().__init__()

        # Configure tokenizers
        self.tokenizers = [
            AutoTokenizer.from_pretrained(tokenizer) for tokenizer in tokenizers
        ]
        for tokenizer in self.tokenizers:
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.model_max_length > max_length:
                tokenizer.model_max_length = max_length

        # Configure text models
        self.text_models = []
        self.configs = []
        self.max_bucket = 0
        self.use_normed_ctx = use_normed_ctx

        for text_model, extra_config in text_model_and_configs:
            self.text_models.append(load_any(text_model))
            if not isinstance(extra_config, TextModelExtraConfig):
                extra_config = TextModelExtraConfig(**extra_config)
            self.configs.append(extra_config)
            self.max_bucket = max(self.max_bucket, extra_config.concat_bucket)

        self.text_models = nn.ModuleList(self.text_models)

        # Other arguments
        self.zero_for_padding = zero_for_padding

    def tokenize(self, text, **kwargs):
        results = []
        for tokenizer in self.tokenizers:
            results.append(tokenizer(text, **kwargs, return_tensors="pt"))
        return results

    def encode(
        self,
        text: list[str],
        pad_to_n_elements: int | None = None,
        text_model_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        return self.forward(self.tokenize(text, **kwargs), **text_model_kwargs)

    def forward(
        self, tokenizers_outputs, **text_model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: torch.Tensor
            normed_embedding: torch.Tensor
            pooled_embedding: torch.Tensor
            attn_mask: torch.Tensor
        """
        attn_masks = [None for _ in range(self.max_bucket + 1)]
        text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        normed_text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        pooled_text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        for idx, (tokens, text_model, config) in enumerate(
            zip(tokenizers_outputs, self.text_models, self.configs)
        ):
            bucket = config.concat_bucket
            need_mask = config.need_mask
            use_pooled = config.use_pooled
            layer_idx = config.layer_idx
            disable_autocast = config.disable_autocast

            input_ids = tokens["input_ids"].to(self.device)
            attn_mask = tokens["attention_mask"].to(self.device)
            if attn_masks[bucket] is None and need_mask:
                attn_masks[bucket] = attn_mask

            with torch.autocast("cuda", enabled=not disable_autocast):
                normed_embedding, pooled_embedding, *embeddings = text_model(
                    input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=False,
                    **text_model_kwargs,
                )
                # The case of CLIP
                if len(embeddings):
                    # embeddings is tuple
                    embedding = embeddings[-1][layer_idx]
                # The case of T5
                else:
                    embedding = pooled_embedding[-1]
                    pooled_embedding = None
                # print(embedding.shape)

                # SD1/SD2 need this
                if isinstance(text_model, CLIPTextModel):
                    normed_embedding = text_model.text_model.final_layer_norm(embedding)

            # Autocast may cause wrong dtype
            embedding = embedding.to(self.dtype)
            normed_embedding = normed_embedding.to(self.dtype)
            pooled_embedding = pooled_embedding.to(self.dtype)

            if self.zero_for_padding:
                n_squeezes = embedding.ndim - attn_mask.ndim
                for _ in range(n_squeezes):
                    attn_mask = attn_mask.unsqueeze(-1)
                embedding = embedding * attn_mask
                normed_embedding = normed_embedding * attn_mask
                for _ in range(n_squeezes):
                    attn_mask = attn_mask.squeeze(-1)
            text_embeddings[bucket].append(embedding)
            normed_text_embeddings[bucket].append(normed_embedding)
            if use_pooled and pooled_embedding is not None:
                pooled_text_embeddings[bucket].append(pooled_embedding)

        for i in range(len(text_embeddings)):
            if text_embeddings[i] == []:
                text_embeddings[i] = None
                normed_text_embeddings[i] = None
                pooled_text_embeddings[i] = None
                continue
            text_embeddings[i] = torch.cat(text_embeddings[i], dim=-1)
            normed_text_embeddings[i] = torch.cat(normed_text_embeddings[i], dim=-1)
            if pooled_text_embeddings[i] == []:
                pooled_text_embeddings[i] = None
                continue
            pooled_text_embeddings[i] = torch.cat(pooled_text_embeddings[i], dim=-1)

        max_dim = max(
            embedding.size(-1) for embedding in text_embeddings if embedding is not None
        )
        for idx, embedding in enumerate(text_embeddings):
            if embedding is None:
                continue
            if embedding.size(-1) < max_dim:
                text_embeddings[idx] = torch.nn.functional.pad(
                    embedding, (0, max_dim - embedding.size(-1))
                )
        for idx, embedding in enumerate(normed_text_embeddings):
            if embedding is None:
                continue
            if embedding.size(-1) < max_dim:
                normed_text_embeddings[idx] = torch.nn.functional.pad(
                    embedding, (0, max_dim - embedding.size(-1))
                )

        if any(mask is not None for mask in attn_masks):
            for idx, embedding in enumerate(text_embeddings):
                if embedding is None:
                    continue
                elif attn_masks[idx] is None:
                    attn_masks[idx] = torch.ones(
                        embedding.size(0), embedding.size(1), device=embedding.device
                    ).long()
            attn_masks = torch.cat(remove_none(attn_masks), dim=1)
        else:
            attn_masks = None

        if any(pooled is not None for pooled in pooled_text_embeddings):
            pooled_text_embeddings = torch.cat(
                remove_none(pooled_text_embeddings), dim=-1
            )
        else:
            pooled_text_embeddings = None

        text_embeddings = torch.cat(remove_none(text_embeddings), dim=1)
        normed_text_embeddings = torch.cat(remove_none(normed_text_embeddings), dim=1)

        return (
            text_embeddings,
            normed_text_embeddings,
            pooled_text_embeddings,
            attn_masks,
        )
