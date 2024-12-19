# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0
from einops import rearrange

from ..utils.aggregation import concat_aggregate_embeddings_vectorize
from .attn_masks import LayerAttnMeta


class CombinedAttnProcessor:
    def __init__(self, self_attn_processor, cross_attn_processor):
        self.self_attn_processor = self_attn_processor
        self.cross_attn_processor = cross_attn_processor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        temb: torch.FloatTensor | None = None,
        n_elements_per_image: list[int] | None = None,
        pad_to_n_elements: int | None = None,
        region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        ip_region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        ip_adapter_scale: float | None = None,
        layer_attn_meta_dict: dict[int, LayerAttnMeta] | None = None,
    ):
        if encoder_hidden_states is None:
            if isinstance(self.self_attn_processor, LayerAttnProcessor):
                return self.self_attn_processor(
                    attn,
                    hidden_states,
                    temb=temb,
                    n_elements_per_image=n_elements_per_image,
                    pad_to_n_elements=pad_to_n_elements,
                    layer_attn_meta_dict=layer_attn_meta_dict,
                )
            return self.self_attn_processor(attn, hidden_states, temb=temb)
        else:
            if isinstance(self.cross_attn_processor, RegionAttnProcessor):
                return self.cross_attn_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    region_mask_dict=region_mask_dict,
                    ip_region_mask_dict=ip_region_mask_dict,
                    ip_adapter_scale=ip_adapter_scale,
                )
            return self.cross_attn_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
            )


def compute_attention_score(
    query: torch.Tensor, key: torch.Tensor, return_logits: bool = True
):
    # Get dimensions
    batch_size, num_heads, query_length, head_dim = query.size()
    _, _, key_length, _ = key.size()
    # Compute raw attention scores
    # Scores shape: [batch_size, num_heads, query_length, key_length]
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(head_dim, dtype=query.dtype)
    )
    if not return_logits:
        scores = F.softmax(scores, dim=-1)
    return scores


class RegionAttnProcessor:
    """
    Region mask-based cross-attention
    Adapted from https://github.com/Birch-san/regional-attn/blob/main/src/attention/regional_attn.py  # noqa
    """

    def __init__(
        self,
        use_flex_attention: bool = True,
        unit_seq_len: int = 77,
    ):
        self.use_flex_attention = use_flex_attention
        self.flex_attention = None
        self.to_k_ip = None
        self.to_v_ip = None
        # in place update of attn score cache
        self.attn_score_cache = None
        self.unit_seq_len = unit_seq_len

    @classmethod
    def from_ip_adapter(
        cls,
        ip_adapter: IPAdapterAttnProcessor2_0,
        ip_adapter_scale: float = 1.0,
        use_flex_attention: bool = True,
        unit_seq_len: int = 77,
    ) -> "RegionAttnProcessor":
        region_attn_processor = cls(
            use_flex_attention=use_flex_attention,
            unit_seq_len=unit_seq_len,
        )
        # We assume there is only one ip adapter
        region_attn_processor.to_k_ip = ip_adapter.to_k_ip[0]
        region_attn_processor.to_v_ip = ip_adapter.to_v_ip[0]
        region_attn_processor.ip_adapter_scale = ip_adapter_scale
        return region_attn_processor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        temb: torch.FloatTensor | None = None,
        region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        ip_region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        ip_adapter_scale: float | None = None,
    ) -> torch.FloatTensor:
        """
        Parameters
        ----------
        attn
            Attention module
        hidden_states
            Input tensor of size [batch_size, sequence_length_q, hidden_size_q]
            This represents the flattened image patches, where sequence_length_q
            is height * width
        encoder_hidden_states
            Text conditioning of size [batch_size, sequence_length_k, hidden_size_k]
            This represents the concatenated text embedding, where sequence_length_k
            is n_captions * max_caption_length
        attention_mask
            Encoder attention mask of size [batch_size, 1, sequence_length_k]
            It takes values in {0, -inf}
        temb
            time embedding
        region_mask_dict
            A dictionary mapping from sequence_length_q to region masks.
            Region mask should be of size
            [batch_size, sequence_length_q, sequence_length_k].
            Region mask should take boolean values.

        Returns
        -------
        torch.FloatTensor
        """
        # The compiled flex attention cannot be pickled
        # To circumvent this we only initialize it in the first call
        if self.use_flex_attention and self.flex_attention is None:
            # dynamic needs to be set to False
            # compilation is needed
            self.flex_attention = torch.compile(flex_attention, dynamic=False)

        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        else:
            ip_hidden_states = None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = rearrange(query, "b n (nh d) -> b nh n d", nh=attn.heads)
        key = rearrange(key, "b n (nh d) -> b nh n d", nh=attn.heads)
        value = rearrange(value, "b n (nh d) -> b nh n d", nh=attn.heads)

        if self.attn_score_cache is not None:
            # If we set return_logits to False, we compute softmax over
            # the concatenation
            score = compute_attention_score(query, key, return_logits=True)
            score = rearrange(
                score,
                "... (n_units unit_seq_len) -> ... n_units unit_seq_len",
                unit_seq_len=self.unit_seq_len,
            )
            score = torch.softmax(score, dim=-1)
            score = rearrange(
                score,
                "... n_units unit_seq_len -> ... (n_units unit_seq_len)",
                unit_seq_len=self.unit_seq_len,
            )
            q_length = hidden_states.shape[1]
            if q_length not in self.attn_score_cache:
                self.attn_score_cache[q_length] = (1, score)
            else:
                self.attn_score_cache[q_length] = (
                    self.attn_score_cache[q_length][0] + 1,
                    self.attn_score_cache[q_length][1] + score,
                )

        if self.use_flex_attention:
            # Ignore attention mask input and assume it is already taken
            # into account in region_mask_dict
            if region_mask_dict is not None:
                attention_mask = region_mask_dict[hidden_states.shape[1]]
            else:
                attention_mask = None
            hidden_states = self.flex_attention(
                query, key, value, block_mask=attention_mask
            )
        else:
            if attention_mask is not None:
                # Convert from {0, -inf} to {True, False}
                attention_mask = (attention_mask == 0).bool()
            if region_mask_dict is not None:
                region_attention_mask = region_mask_dict[hidden_states.shape[1]].bool()
                if attention_mask is None:
                    attention_mask = region_attention_mask
                else:
                    attention_mask = torch.logical_and(
                        attention_mask, region_attention_mask
                    )
            # Add head dimension
            if attention_mask is not None and attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        hidden_states = rearrange(hidden_states, "b nh n d -> b n (nh d)")

        if ip_hidden_states is not None:
            ip_adapter_scale = ip_adapter_scale or self.ip_adapter_scale
            if ip_adapter_scale > 0.0:
                assert ip_region_mask_dict is not None
                assert self.to_k_ip is not None
                assert self.to_v_ip is not None
                # seems that model.to is not working for attention processors
                if self.to_k_ip.weight.device != hidden_states.device:
                    self.to_k_ip = self.to_k_ip.to(hidden_states)
                if self.to_v_ip.weight.device != hidden_states.device:
                    self.to_v_ip = self.to_v_ip.to(hidden_states)
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)
                ip_key = rearrange(ip_key, "b n (nh d) -> b nh n d", nh=attn.heads)
                ip_value = rearrange(ip_value, "b n (nh d) -> b nh n d", nh=attn.heads)
                attention_mask = ip_region_mask_dict[hidden_states.shape[1]]
                # For simplicity, never use flex attention for ip_hidden_states
                # Add head dimension
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(1)
                    # Identify queries that attend to at least one key/value
                    # Shape: [batch_size, 1, query_len]
                    attention_mask_q = torch.any(attention_mask, dim=-1)
                ip_hidden_states = F.scaled_dot_product_attention(
                    query,
                    ip_key,
                    ip_value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                ip_hidden_states = rearrange(ip_hidden_states, "b nh n d -> b n (nh d)")
                if attention_mask is not None:
                    # Squeeze and expand dimensions to match ip_hidden_states
                    attention_mask_q = attention_mask_q.squeeze(1).unsqueeze(
                        -1
                    )  # Shape: [batch_size, query_len, 1]
                    # Zero out ip_hidden_states where queries do not
                    # attend to any keys/values
                    ip_hidden_states = (
                        ip_hidden_states * attention_mask_q
                    )  # Element-wise multiplication
                hidden_states = hidden_states + ip_adapter_scale * ip_hidden_states

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LayerAttnProcessor:
    def __init__(self, use_flex_attention: bool = True):
        self.use_flex_attention = use_flex_attention
        self.flex_attention = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        n_elements_per_image: torch.LongTensor,
        layer_attn_meta_dict: dict[int, LayerAttnMeta] | None,
        pad_to_n_elements: int | None = None,
        temb: torch.FloatTensor | None = None,
    ):
        # The compiled flex attention cannot be pickled
        # To circumvent this we only initialize it in the first call
        if self.use_flex_attention and self.flex_attention is None:
            # dynamic needs to be set to False
            # compilation is needed
            self.flex_attention = torch.compile(flex_attention, dynamic=False)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        batch_size, sequence_length, _ = hidden_states.shape

        if layer_attn_meta_dict is None:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = rearrange(query, "b n (nh d) -> b nh n d", nh=attn.heads)
            key = rearrange(key, "b n (nh d) -> b nh n d", nh=attn.heads)
            value = rearrange(value, "b n (nh d) -> b nh n d", nh=attn.heads)

            hidden_states = F.scaled_dot_product_attention(query, key, value)

        else:
            meta = layer_attn_meta_dict[sequence_length]

            if not isinstance(n_elements_per_image, torch.Tensor):
                n_elements_per_image = torch.tensor(n_elements_per_image)

            # Put layers of the same image together, which affects sequence_length
            hidden_states_agg = concat_aggregate_embeddings_vectorize(
                hidden_states,
                n_elements_per_image,
                pad_to_n_elements=pad_to_n_elements,
                batch_indices_flat=meta.agg_batch_indices_flat,
                positions_flat=meta.agg_positions_flat,
                cat_embeddings=meta.cat_embeddings,
            )
            # Allocate memory for the concatenated embedding
            if meta.cat_embeddings is None:
                meta.cat_embeddings = hidden_states_agg
            n_elements_per_image = n_elements_per_image.to(hidden_states.device)

            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states_agg)
            value = attn.to_v(hidden_states_agg)

            # Adjustment with context length
            # key = key / torch.sqrt(n_elements_per_image).to(key).unsqueeze(-1).unsqueeze(-1)  # noqa
            key = key.repeat_interleave(n_elements_per_image, dim=0)
            value = value.repeat_interleave(n_elements_per_image, dim=0)

            query = rearrange(query, "b n (nh d) -> b nh n d", nh=attn.heads)
            key = rearrange(key, "b n (nh d) -> b nh n d", nh=attn.heads)
            value = rearrange(value, "b n (nh d) -> b nh n d", nh=attn.heads)

            attention_mask = meta.layer_attn_mask

            # Note that providing all True mask and not providing mask
            # give different results
            if self.use_flex_attention:
                hidden_states = self.flex_attention(
                    query, key, value, block_mask=attention_mask
                )
            else:
                # Add head dimension
                # print(query.shape, key.shape, value.shape, attention_mask.shape)
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask.unsqueeze(1)
                )

        hidden_states = rearrange(hidden_states, "b nh n d -> b n (nh d)")

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
