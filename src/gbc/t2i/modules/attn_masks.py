# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from gbc.t2i.utils.aggregation import (
    get_batch_and_position_indices_for_concat_aggregate,
)


Mask = torch.BoolTensor | BlockMask


def convert_and_stretch_mask(
    attn_mask: torch.BoolTensor,
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = False,
) -> Mask:
    # print(attn_mask.shape)
    if use_flex_attention:
        attn_mask = convert_to_flex_attn_mask(
            attn_mask,
            sequence_length=sequence_length,
            encoder_attn_mask=encoder_attn_mask,
        )
    else:
        attn_mask = attn_mask.repeat_interleave(sequence_length, dim=-1)
        if encoder_attn_mask is not None:
            attn_mask = torch.logical_and(attn_mask, encoder_attn_mask)
    return attn_mask


def convert_to_flex_attn_mask(
    mask: torch.BoolTensor,
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
) -> BlockMask:
    """
    Converts a binary mask to a flex attention mask
    """

    # Must be bool for flex attention
    mask = mask.bool()
    if encoder_attn_mask is not None:
        # remove query dimension
        encoder_attn_mask = encoder_attn_mask.bool().squeeze(1)

    # must be multiple of 128
    kv_length = sequence_length * mask.size(2)
    final_kv_length = math.ceil(kv_length / 128) * 128
    if encoder_attn_mask is not None:
        encoder_attn_mask = F.pad(encoder_attn_mask, (0, final_kv_length - kv_length))

    to_pad_kv = math.ceil((final_kv_length - kv_length) / sequence_length)
    mask = F.pad(mask, (0, to_pad_kv))

    def mask_value(b, h, q_idx, kv_idx):
        mask_value = mask[b, q_idx, kv_idx // sequence_length]
        if encoder_attn_mask is not None:
            mask_value = mask_value & encoder_attn_mask[b, kv_idx]
        return mask_value

    block_mask = create_block_mask(
        mask_value,
        B=mask.size(0),
        H=None,
        Q_LEN=mask.size(1),
        KV_LEN=final_kv_length,
        device=mask.device,
        _compile=True,
    )
    return block_mask


# For region-attention


def convert_mask_dict(
    mask_dict: dict[int, torch.BoolTensor],
    sequence_length: int,
    encoder_attn_mask: torch.BoolTensor | None = None,
    use_flex_attention: bool = False,
) -> dict[int, torch.BoolTensor] | dict[int, BlockMask]:
    return {
        key: convert_and_stretch_mask(
            mask,
            sequence_length,
            encoder_attn_mask,
            use_flex_attention=use_flex_attention,
        )
        for key, mask in mask_dict.items()
    }


# For layer-attention


@dataclass
class LayerAttnMeta:
    # For embedding aggregation
    agg_batch_indices_flat: torch.LongTensor
    agg_positions_flat: torch.LongTensor
    # mask of size
    # [n_total_bboxes, height*width, max_n_bboxes*height*width]
    layer_attn_mask: Mask
    # [n_total_bboxes, max_n_bboxes*height*width]
    attend_to_attn_mask: torch.BoolTensor | None = None
    # Memory allocation for the concatenated embedding
    # [batch_size, max_n_bboxes*height*width, *embedding_dim[2:]]
    # It should be cached within attention layers
    cat_embeddings: torch.FloatTensor | None = None
    converted: bool = False


@dataclass
class LayerMaskConfig:
    # Whether to attend between edges only or not
    use_edge: bool = True
    # For parent to child attention
    attend_from_bbox: bool = True
    # For child to parent attention
    include_child_to_parent: bool = False
    attend_to_bbox: bool = True
    include_diagonal: bool = True


def convert_layer_attn_meta_dict(
    layer_attn_mask_dict: dict[int, LayerAttnMeta],
    use_flex_attention: bool = False,
) -> dict[int, torch.BoolTensor] | dict[int, BlockMask]:
    # This performs in-place operation
    for latent_size, meta in layer_attn_mask_dict.items():
        if not meta.converted:
            meta.layer_attn_mask = convert_and_stretch_mask(
                meta.layer_attn_mask,
                sequence_length=latent_size,
                encoder_attn_mask=meta.attend_to_attn_mask,
                use_flex_attention=use_flex_attention,
            )
            meta.converted = True
    return layer_attn_mask_dict


def make_layer_attn_meta_dict(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    latent_width: int,
    latent_height: int,
    n_levels: int,
    layer_mask_config: LayerMaskConfig = LayerMaskConfig(),
    num_generated_per_image: list[int] | None = None,
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
    # For sdxl, there is no attention at first down block
    skip_top_level: bool = False,
) -> dict[int, LayerAttnMeta]:

    layer_attn_meta_dict = {}
    if num_generated_per_image is None:
        num_generated_per_image = torch.tensor([b.shape[0] for b in bboxes])
    start_idx = 1 if skip_top_level else 0

    for idx in range(n_levels):
        if idx >= start_idx:
            attn_mask, to_attn_mask = get_layer_attention_mask(
                bboxes,
                adjacency,
                latent_width,
                latent_height,
                self_attention=True,
                pad_to_n_bboxes=pad_to_n_bboxes,
                pad_to_n_total_bboxes=pad_to_n_total_bboxes,
                num_generated_per_image=num_generated_per_image,
                layer_mask_config=layer_mask_config,
            )
            to_attn_mask = to_attn_mask.unsqueeze(1)
            size = latent_width * latent_height
            converted = False
            (
                batch_indices_flat,
                positions_flat,
            ) = get_batch_and_position_indices_for_concat_aggregate(
                num_generated_per_image, size
            )
            meta = LayerAttnMeta(
                agg_batch_indices_flat=batch_indices_flat,
                agg_positions_flat=positions_flat,
                layer_attn_mask=attn_mask,
                attend_to_attn_mask=to_attn_mask,
                converted=converted,
            )
            layer_attn_meta_dict[size] = meta
        latent_width = latent_width // 2
        latent_height = latent_height // 2
    return layer_attn_meta_dict


def get_layer_mask_per_image(
    bboxes: torch.Tensor,
    adjacency: list[list[int]],
    width: int,
    height: int,
    layer_mask_config: LayerMaskConfig,
    exclusive_attention: bool = False,
) -> torch.BoolTensor:
    """
    Parameters
    ----------
    bboxes
        A tensor of shape [n_bboxes, 4] representing the bounding boxes

    Returns
    -------
    The mask is of size [n_bboxes, height x width, n_bboxes]
    """

    device = bboxes.device
    n_bboxes = bboxes.shape[0]
    # For parent to child attention
    mask_attend_from = torch.zeros(
        [n_bboxes, height, width, n_bboxes], dtype=torch.bool
    ).to(device)
    mask_attend_to = torch.zeros(
        [n_bboxes, n_bboxes, height, width], dtype=torch.bool
    ).to(device)

    # Attention within each image layer
    if layer_mask_config.include_diagonal:
        for i in range(len(mask_attend_from)):
            mask_attend_from[i, :, :, i] = True
            mask_attend_to[i, i, :, :] = True

    # get relative bbox for each edge
    # source is query (larger image) and target is key (smaller image)
    relative_bboxes_source = []
    relative_bboxes_target = []
    for source, targets in enumerate(adjacency):
        for target in targets:
            # Attend from bbox
            relative_bbox_source = convert_to_relative_bbox(
                bboxes[source], bboxes[target]
            )
            # Attend to bbox
            relative_bbox_target = convert_to_relative_bbox(
                bboxes[target], bboxes[source]
            )
            relative_bboxes_source.append(relative_bbox_source)
            relative_bboxes_target.append(relative_bbox_target)
    relative_bboxes_source = torch.tensor(relative_bboxes_source)
    relative_bboxes_target = torch.tensor(relative_bboxes_target)
    edge_masks_source = bboxes_to_mask(relative_bboxes_source, width, height)
    edge_masks_target = bboxes_to_mask(relative_bboxes_target, width, height)

    edge_counter = 0
    for source, targets in enumerate(adjacency):
        # Decide the nodes that are ancestors of a descendants but descendant of source
        # This works because the adjacency actually points to all the descendants
        # when we use this function for cross-attention
        if exclusive_attention:
            ancestors = {des: [source] for des in targets}
            for target in targets:
                for target_descendants in adjacency[target]:
                    ancestors[target_descendants].append(target)
        for target in targets:
            if layer_mask_config.attend_from_bbox:
                mask_attend_from[source, :, :, target] = edge_masks_source[edge_counter]
                if exclusive_attention:
                    for anc in ancestors[target]:
                        mask_attend_from[source, :, :, anc][
                            edge_masks_source[edge_counter]
                        ] = False
                if layer_mask_config.include_child_to_parent:
                    mask_attend_from[target, :, :, source] = edge_masks_target[
                        edge_counter
                    ]
            else:
                mask_attend_from[source, :, :, target] = True
                if layer_mask_config.include_child_to_parent:
                    mask_attend_from[target, :, :, source] = True
            if layer_mask_config.attend_to_bbox:
                mask_attend_to[source, target, :, :] = edge_masks_target[edge_counter]
                if layer_mask_config.include_child_to_parent:
                    mask_attend_to[target, source, :, :] = edge_masks_source[
                        edge_counter
                    ]
            else:
                mask_attend_to[source, target, :, :] = True
                if layer_mask_config.include_child_to_parent:
                    mask_attend_to[target, source, :, :] = True
            edge_counter += 1

    return mask_attend_from.flatten(start_dim=1, end_dim=2), mask_attend_to.flatten(
        start_dim=1, end_dim=3
    )


def get_layer_attention_mask(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    width: int,
    height: int,
    self_attention: bool = True,
    exclusive_attention: bool = False,
    num_generated_per_image: list[int] | None = None,  # Used for cross-attn
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
    layer_mask_config: LayerMaskConfig | None = None,
) -> torch.BoolTensor:
    """
    Parameters
    ----------
    bboxes
        A list of tensors of shape [n_bboxes, 4] representing the bounding boxes
    adjacency
        A list of adjacency lists. Each adjacency list is a list of lists representing
        the adjacency of the corresponding graph.

    Returns
    -------
    The mask is of size
    [n_total_bboxes, height x width, n_bboxes]
    """
    if num_generated_per_image is None:
        num_generated_per_image = [len(bbox) for bbox in bboxes]
    if self_attention:
        max_n_bboxes = pad_to_n_bboxes or max(num_generated_per_image)
    # cross-attention
    else:
        max_n_bboxes = pad_to_n_bboxes or max(len(bbox) for bbox in bboxes)
    n_total_generated = pad_to_n_total_bboxes or sum(num_generated_per_image)
    device = bboxes[0].device
    layer_mask_config = layer_mask_config or LayerMaskConfig()

    mask_attend_from = torch.zeros(
        [n_total_generated, height * width, max_n_bboxes], dtype=torch.bool
    ).to(device)
    mask_attend_to = torch.zeros(
        [n_total_generated, max_n_bboxes * height * width], dtype=torch.bool
    ).to(device)

    bbox_start = 0
    for i, (bboxes_i, num_generated_i) in enumerate(
        zip(bboxes, num_generated_per_image)
    ):
        num_attend_to = num_generated_i if self_attention else len(bboxes_i)
        if not layer_mask_config.use_edge:
            # Allow all the layers within the same image to attend to each other
            mask_attend_from[
                bbox_start : bbox_start + num_generated_i, :, :num_attend_to
            ] = True
            mask_attend_to[
                bbox_start : bbox_start + num_generated_i,
                : num_attend_to * height * width,
            ] = True
        else:
            # Only allow attention via edges
            mask_from_per_image, mask_to_per_image = get_layer_mask_per_image(
                bboxes_i,
                adjacency[i],
                width,
                height,
                layer_mask_config=layer_mask_config,
                exclusive_attention=exclusive_attention,
            )
            mask_attend_from[
                bbox_start : bbox_start + num_generated_i, :, :num_attend_to
            ] = mask_from_per_image[:num_generated_i, :, :num_attend_to]
            sequence_length_i = num_attend_to * height * width
            mask_attend_to[
                bbox_start : bbox_start + num_generated_i, :sequence_length_i
            ] = mask_to_per_image[:num_generated_i, :sequence_length_i]
        bbox_start += num_generated_i

    return mask_attend_from, mask_attend_to


# For layer-region-attention


def make_layer_region_mask_dict(
    bboxes: list[torch.Tensor],
    adjacency: list[list[list[int]]],
    latent_width: int,
    latent_height: int,
    n_levels: int,
    use_adj: bool = False,  # by default we use descendants
    include_diagonal: bool = True,
    num_generated_per_image: list[int] | None = None,
    pad_to_n_bboxes: int | None = None,
    pad_to_n_total_bboxes: int | None = None,
    # For sdxl, there is no attention at first down block
    skip_top_level: bool = False,
    exclusive_attention: bool = False,
) -> dict[int, Mask]:
    """
    Converts a set of relative bounding boxes and adjacency lists to a dictionary
    mapping feature map size to binary mask of size
    [n_total_bboxes, feature_height x feature_width, max_n_bboxes]
    """

    layer_attn_meta_dict = {}
    if use_adj:
        descendants = adjacency
    else:
        descendants = [get_descendants(adj) for adj in adjacency]
    layer_mask_config = LayerMaskConfig(include_diagonal=include_diagonal)
    start_idx = 1 if skip_top_level else 0

    for idx in range(n_levels):
        if idx >= start_idx:
            # b, q, n_bboxes
            attn_mask = get_layer_attention_mask(
                bboxes,
                descendants,
                latent_width,
                latent_height,
                self_attention=False,
                exclusive_attention=exclusive_attention,
                num_generated_per_image=num_generated_per_image,
                layer_mask_config=layer_mask_config,
                pad_to_n_bboxes=pad_to_n_bboxes,
                pad_to_n_total_bboxes=pad_to_n_total_bboxes,
            )[0]
            size = latent_width * latent_height
            layer_attn_meta_dict[size] = attn_mask
        latent_width = latent_width // 2
        latent_height = latent_height // 2
    return layer_attn_meta_dict


def get_descendants(adjacency: list[list[int]]) -> list[list[int]]:
    """
    This only works for DAG
    """
    descendants = [None] * len(adjacency)

    def dfs(node):
        if descendants[node] is not None:
            return descendants[node]
        descendants[node] = set()
        for child in adjacency[node]:
            descendants[node].update(dfs(child))
            descendants[node].add(child)
        # descendants[node].add(node)
        return descendants[node]

    for node in range(len(adjacency)):
        dfs(node)

    descendants = [sorted(x) for x in descendants]
    return descendants


# bbox utilities


Bbox = tuple[float, float, float, float]


def convert_to_relative_bbox(outer_bbox: Bbox, inner_bbox_global: Bbox) -> Bbox:
    """
    Converts a bounding box (inner_bbox_global) that is relative to
    the entire image into coordinates that are relative to
    another bounding box (outer_bbox).

    Args:
        outer_bbox:
            The outer bounding box relative to the entire image.
        inner_bbox_global:
            The inner bounding box relative to the entire image.

    Returns:
        A new Bbox instance representing the inner bounding box
        relative to the outer bounding box.
    """
    inner_left, inner_top, inner_right, inner_bottom = inner_bbox_global
    outer_left, outer_top, outer_right, outer_bottom = outer_bbox
    if outer_right <= outer_left or outer_bottom <= outer_top:
        relative_left = 0.0
        relative_top = 0.0
        relative_right = 0.0
        relative_bottom = 0.0
    else:
        relative_left = (inner_left - outer_left) / (outer_right - outer_left)
        relative_top = (inner_top - outer_top) / (outer_bottom - outer_top)
        relative_right = (inner_right - outer_left) / (outer_right - outer_left)
        relative_bottom = (inner_bottom - outer_top) / (outer_bottom - outer_top)
    # inner bbox is not necessarily fully contained in outer bbox
    relative_left = max(0.0, min(1.0, relative_left))
    relative_top = max(0.0, min(1.0, relative_top))
    relative_right = max(0.0, min(1.0, relative_right))
    relative_bottom = max(0.0, min(1.0, relative_bottom))
    return relative_left, relative_top, relative_right, relative_bottom


def bboxes_to_mask(
    bboxes: torch.Tensor, target_width: int, target_height: int
) -> torch.BoolTensor:
    """
    Converts a set of relative bounding boxes to a binary mask of size
    [n_bboxes, target_height, target_width].

    Parameters
    ----------
    bboxes : torch.Tensor
        A tensor of shape [n_bboxes, 4] where each bounding box is represented
        by four relative coordinates [left, top, right, bottom].
        Each coordinate is a relative value between 0 and 1,
        where 0 corresponds to the left/top and 1 corresponds to the right/bottom.
    target_width : int
        The desired width of the output mask.
    target_height : int
        The desired height of the output mask.

    Returns
    -------
    torch.Tensor
        A tensor of shape [n_bboxes, target_height, target_width],
        where the regions corresponding to each bounding box are set to True.
        The rest of the mask is set to False.

    Example
    -------
    >>> bboxes = torch.tensor([[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]])
    >>> result = bboxes_to_mask(bboxes, 100, 100)
    >>> result.shape
    torch.Size([2, 100, 100])
    """
    n_bboxes = bboxes.shape[0]
    mask = bboxes.new_zeros((n_bboxes, target_height, target_width)).bool()

    for i, bbox in enumerate(bboxes):
        left = int(target_width * bbox[0])
        top = int(target_height * bbox[1])
        right = int(target_width * bbox[2])
        bottom = int(target_height * bbox[3])

        if right > left and bottom > top:
            mask[i, top:bottom, left:right] = True

    return mask


if __name__ == "__main__":

    bboxes = [
        torch.tensor([[0, 0, 1, 1], [0.3, 0.5, 0.7, 1]]),
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0.5], [0.5, 0.5, 1, 1]]),
    ]
    adjacency = [[[1], []], [[1, 2], [], []]]
    mask_from, mask_to = get_layer_attention_mask(bboxes, adjacency, 2, 2)
    print(mask_from)
    print(mask_from.shape)
    print(mask_to)
    print(mask_to.shape)

    mask = make_layer_region_mask_dict(bboxes, adjacency, 8, 8, n_levels=2)
    print(mask)
    print(mask[64].shape)
