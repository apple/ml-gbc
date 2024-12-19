# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import inspect
import copy

import torch
from diffusers import UNet2DConditionModel
from diffusers.utils.hub_utils import _get_model_file
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import load_state_dict
from diffusers.models.attention_processor import AttnProcessor2_0

from gbc.utils import get_gbc_logger
from gbc.t2i.utils.aggregation import aggregate_embeddings
from gbc.t2i.modules.attn_masks import (
    convert_mask_dict,
    convert_layer_attn_meta_dict,
    LayerAttnMeta,
)
from gbc.t2i.modules.attention import (
    CombinedAttnProcessor,
    RegionAttnProcessor,
    LayerAttnProcessor,
)
from gbc.t2i.modules.graph_attn import GraphAttnMetaPerImage


class UNet2DConditionModelAllowOverride(UNet2DConditionModel):
    # https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/configuration_utils.py#L452  # noqa
    # The issue comes from the use of `extract_init_dict` in from_config
    @staticmethod
    def _get_init_keys(input_class):
        input_class_params = set(
            inspect.signature(input_class.__init__).parameters.keys()
        )
        if issubclass(input_class, UNet2DConditionModelAllowOverride):
            for parent_class in input_class.__bases__:
                input_class_params |= input_class._get_init_keys(parent_class)
        return input_class_params


class UNet2DConditionModelSetAttnProcessor(UNet2DConditionModelAllowOverride):
    @register_to_config
    def __init__(self, *args, attn_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        if attn_processor is not None:
            self.set_attn_processor(attn_processor)


class UNet2DConditionModelWithIPAdapter(UNet2DConditionModelAllowOverride):

    @classmethod
    def from_pretrained(cls, *args, load_ip_adapter_kwargs={}, **kwargs):
        unet = super().from_pretrained(*args, **kwargs)
        unet.load_ip_adapter(**load_ip_adapter_kwargs)
        return unet

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path: str = "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name: str = "ip-adapter_sdxl.bin",
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        model_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=weight_name,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        if weight_name.endswith(".safetensors"):
            from safetensors import safe_open

            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = (
                            f.get_tensor(key)
                        )
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                            f.get_tensor(key)
                        )
        else:
            state_dict = load_state_dict(model_file)

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            raise ValueError(
                "Required keys are (`image_proj` and `ip_adapter`) "
                "missing from the state dict."
            )
        self._load_ip_adapter_weights([state_dict])


class GbcUNet2DConditionModel(UNet2DConditionModelSetAttnProcessor):

    load_ip_adapter = UNet2DConditionModelWithIPAdapter.load_ip_adapter

    @register_to_config
    def __init__(
        self,
        *args,
        use_region_attention: bool = True,
        use_layer_attention: bool = True,
        use_flex_attention_for_region: bool = True,
        use_flex_attention_for_layer: bool = True,
        skip_set_attn_processor: bool = False,
        only_first_pool: bool = False,
        use_caption_mask: bool = False,
        **kwargs,
    ):
        # https://github.com/pytorch/pytorch/issues/104674
        if use_flex_attention_for_region or use_flex_attention_for_layer:
            torch._dynamo.config.optimize_ddp = False
        if skip_set_attn_processor:
            attn_processor = None
        else:
            self_attn_processor = (
                LayerAttnProcessor(
                    use_flex_attention=use_flex_attention_for_layer,
                )
                if use_layer_attention
                else AttnProcessor2_0()
            )
            if use_region_attention:
                cross_attn_processor = RegionAttnProcessor(
                    use_flex_attention=use_flex_attention_for_region,
                )
            else:
                cross_attn_processor = AttnProcessor2_0()
            attn_processor = CombinedAttnProcessor(
                self_attn_processor, cross_attn_processor
            )
        super().__init__(*args, attn_processor=attn_processor, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        *args,
        use_region_attention: bool = True,
        use_layer_attention: bool = True,
        use_flex_attention_for_region: bool = True,
        use_flex_attention_for_layer: bool = True,
        load_ip_adapter_kwargs: dict = {},
        ip_adapter_scale: float = 0.0,
        **kwargs,
    ):
        unet = super().from_pretrained(
            *args,
            use_region_attention=use_region_attention,
            use_layer_attention=use_layer_attention,
            use_flex_attention_for_region=use_flex_attention_for_region,
            use_flex_attention_for_layer=use_flex_attention_for_layer,
            skip_set_attn_processor=ip_adapter_scale > 0.0,
            **kwargs,
        )
        if ip_adapter_scale > 0.0:
            logger = get_gbc_logger()
            logger.info("Load IP adapter")
            unet.load_ip_adapter(**load_ip_adapter_kwargs)
            attn_procs = {}
            for name, attn_proc in unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    attn_proc = CombinedAttnProcessor(
                        self_attn_processor=(
                            LayerAttnProcessor(
                                use_flex_attention=use_flex_attention_for_layer,
                            )
                            if use_layer_attention
                            else AttnProcessor2_0()
                        ),
                        cross_attn_processor=AttnProcessor2_0(),
                    )
                else:
                    if use_region_attention:
                        cross_attn_processor = RegionAttnProcessor.from_ip_adapter(
                            attn_proc,
                            ip_adapter_scale=ip_adapter_scale,
                            use_flex_attention=use_flex_attention_for_region,
                        )
                    else:
                        cross_attn_processor = AttnProcessor2_0()
                    attn_proc = CombinedAttnProcessor(
                        self_attn_processor=AttnProcessor2_0(),
                        cross_attn_processor=cross_attn_processor,
                    )
                attn_procs[name] = attn_proc
            unet.set_attn_processor(attn_procs)
            # deal with hid proj in forward below directly
            unet.config.encoder_hid_dim_type = None
        return unet

    def set_xattn_score_cache(self):
        """
        Set attention score cache to store attention scores
        """
        self.xattn_score_cache = {}
        for attn_proc in self.attn_processors.values():
            # This is because we use CombinedAttnProcessor
            if hasattr(attn_proc.cross_attn_processor, "attn_score_cache"):
                attn_proc.cross_attn_processor.attn_score_cache = self.xattn_score_cache

    def get_xattn_score_cache(self):
        return self.xattn_score_cache

    def clear_xattn_score_cache(self):
        self.xattn_score_cache = None
        for attn_proc in self.attn_processors.values():
            if hasattr(attn_proc.cross_attn_processor, "attn_score_cache"):
                attn_proc.cross_attn_processor.attn_score_cache = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        added_cond_kwargs: dict | None = None,
        cross_attention_kwargs: dict | None = None,
        # specific for layer and region attention
        n_elements_per_image: torch.Tensor | None = None,
        n_generated_per_image: list[int] | None = None,
        pad_to_n_elements: int | None = None,
        region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        ip_region_mask_dict: dict[int, torch.BoolTensor] | None = None,
        # A mask to indicate if reference image is used for a prompt or not
        # of size [b]
        ip_reference_presence_mask: torch.BoolTensor | None = None,
        layer_attn_meta_dict: dict[int, LayerAttnMeta] | None = None,
        graph_attn_meta: list[GraphAttnMetaPerImage] | None = None,
        use_caption_mask: bool | None = None,
        **kwargs,
    ):
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        else:
            cross_attention_kwargs = copy.copy(cross_attention_kwargs)
        # This avoids modifying the original added_cond_kwargs
        # Needed for sampling
        added_cond_kwargs = copy.copy(added_cond_kwargs)
        if pad_to_n_elements is not None:
            cross_attention_kwargs["pad_to_n_elements"] = pad_to_n_elements

        if self.config.use_layer_attention and n_generated_per_image is None:
            n_generated_per_image = n_elements_per_image
        if n_generated_per_image is not None:
            cross_attention_kwargs["n_elements_per_image"] = n_generated_per_image

        if "image_embeds" in added_cond_kwargs:
            # encoder_hid_proj is MultiIPAdapterImageProjection
            # each image_projection_layer is ImageProjection
            # [b, 1, 1280] -> [b, 1, 4, 1028] -> [b, 4, 1028]
            # The second dimension is the "number of ref images" dimension
            image_embeds = self.encoder_hid_proj(added_cond_kwargs["image_embeds"])[0]
            image_embeds = image_embeds.squeeze(1)
            img_sequence_length = image_embeds.size(1)
        else:
            image_embeds = None
        sequence_length = encoder_hidden_states.size(1)

        if self.config.use_region_attention:
            assert n_elements_per_image is not None
            prompt_idx = 0
            # Shape (n_prompts, unit_sequence_length)
            caption_mask = torch.ones(
                *encoder_hidden_states.shape[:2],
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )
            adjacencies = []
            for n_prompts_i, graph_attn_meta_i in zip(
                n_elements_per_image, graph_attn_meta
            ):
                if graph_attn_meta_i is None:
                    adjacencies.append(None)
                    prompt_idx += n_prompts_i
                    continue
                caption_mask[prompt_idx : prompt_idx + n_prompts_i] = (
                    graph_attn_meta_i.caption_mask.to(caption_mask)
                )
                adjacencies.append(graph_attn_meta_i.get_adjacency())
                prompt_idx += n_prompts_i

            if use_caption_mask is None:
                use_caption_mask = self.config.use_caption_mask
            if use_caption_mask:
                # Combine caption mask with tokenizer mask
                if encoder_attention_mask is None:
                    encoder_attention_mask = caption_mask
                else:
                    encoder_attention_mask = torch.logical_and(
                        encoder_attention_mask, caption_mask
                    )

            encoder_hidden_states = aggregate_embeddings(
                encoder_hidden_states,
                n_elements_per_image,
                mode="concat",
                pad_to_n_elements=pad_to_n_elements,
            )
            if image_embeds is not None:
                image_embeds = aggregate_embeddings(
                    image_embeds,
                    n_elements_per_image,
                    mode="concat",
                    pad_to_n_elements=pad_to_n_elements,
                )
            if encoder_attention_mask is not None:
                encoder_attention_mask = aggregate_embeddings(
                    encoder_attention_mask,
                    n_elements_per_image,
                    mode="concat",
                    pad_to_n_elements=pad_to_n_elements,
                )
            if n_generated_per_image is not None:
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                    n_generated_per_image, dim=0
                )
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.repeat_interleave(
                        n_generated_per_image, dim=0
                    )
                if image_embeds is not None:
                    image_embeds = image_embeds.repeat_interleave(
                        n_generated_per_image, dim=0
                    )
                if "text_embeds" in added_cond_kwargs:
                    if self.config.only_first_pool:
                        text_embeds = aggregate_embeddings(
                            added_cond_kwargs["text_embeds"],
                            n_elements_per_image,
                            mode="first",
                        )
                        text_embeds = text_embeds.repeat_interleave(
                            n_generated_per_image, dim=0
                        )
                    else:
                        text_embeds = aggregate_embeddings(
                            added_cond_kwargs["text_embeds"],
                            n_elements_per_image,
                            mode="first",
                            n_generated_per_image=n_generated_per_image,
                        )
                    added_cond_kwargs["text_embeds"] = text_embeds
            elif "text_embeds" in added_cond_kwargs:
                added_cond_kwargs["text_embeds"] = aggregate_embeddings(
                    added_cond_kwargs["text_embeds"], n_elements_per_image, mode="first"
                )

            # Prepare region mask
            if encoder_attention_mask is not None:
                encoder_attn_mask = encoder_attention_mask.unsqueeze(1)
            else:
                encoder_attn_mask = None
            region_mask_dict = convert_mask_dict(
                region_mask_dict,
                sequence_length=sequence_length,
                encoder_attn_mask=encoder_attn_mask,
                use_flex_attention=self.config.use_flex_attention_for_region,
            )
            cross_attention_kwargs["region_mask_dict"] = region_mask_dict

            if image_embeds is not None:
                ip_region_mask_dict = ip_region_mask_dict or region_mask_dict
                if ip_reference_presence_mask is not None:
                    ip_encoder_attn_mask = ip_reference_presence_mask.view(-1, 1).tile(
                        [1, img_sequence_length]
                    )
                    ip_encoder_attn_mask = aggregate_embeddings(
                        ip_encoder_attn_mask,
                        n_elements_per_image,
                        mode="concat",
                        pad_to_n_elements=pad_to_n_elements,
                    )
                    if n_generated_per_image is not None:
                        ip_encoder_attn_mask = ip_encoder_attn_mask.repeat_interleave(
                            n_generated_per_image, dim=0
                        )
                    # Add head dimension
                    ip_encoder_attn_mask = ip_encoder_attn_mask.unsqueeze(1)
                else:
                    ip_encoder_attn_mask = None
                ip_region_mask_dict = convert_mask_dict(
                    ip_region_mask_dict,
                    sequence_length=img_sequence_length,
                    encoder_attn_mask=ip_encoder_attn_mask,
                    # Never use flex attention for ip adapter
                    use_flex_attention=False,
                )
                cross_attention_kwargs["ip_region_mask_dict"] = ip_region_mask_dict

        if self.config.use_layer_attention:
            if layer_attn_meta_dict is not None:
                layer_attn_meta_dict = convert_layer_attn_meta_dict(
                    layer_attn_meta_dict,
                    use_flex_attention=self.config.use_flex_attention_for_layer,
                )
            cross_attention_kwargs["layer_attn_meta_dict"] = layer_attn_meta_dict

        if image_embeds is not None:
            encoder_hidden_states = encoder_hidden_states, image_embeds

        return super().forward(
            sample,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        )


if __name__ == "__main__":

    unet = UNet2DConditionModelWithIPAdapter.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        load_ip_adapter_kwargs={
            "pretrained_model_name_or_path": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip-adapter_sdxl.bin",
        },
    )
    print(unet)

    unet = GbcUNet2DConditionModel.from_config(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
    )
    print(unet)
