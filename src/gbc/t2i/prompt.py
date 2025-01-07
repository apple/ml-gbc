# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import re
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from PIL import Image
from omegaconf import ListConfig

import numpy as np
import torch
from torchvision import transforms as T
from transformers import (
    PreTrainedTokenizerBase,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

from gbc.data import Bbox, GbcGraph
from gbc.data.bbox.annotate import annotate_all_labels
from gbc.utils import load_list_from_file
from gbc.t2i.modules.graph_attn import GraphAttnMetaPerImage


class Prompt(ABC):
    pass


Prompt.register(str)


@dataclass
class GbcPrompt(Prompt):
    prompts: list[str]
    bboxes: list[tuple[float, float, float, float]] | torch.Tensor
    adjacency: list[list[int]]
    # For labeled adjacency
    labeled_adjacency: list[list[tuple[int | None, list[str]]]] | None = None
    converted_labeled_adjacency: GraphAttnMetaPerImage | None = None
    # For reference image
    ref_images: list[str | Image.Image | torch.Tensor] | None = None
    ref_image_tensor_dict: dict[str, torch.Tensor] | None = None
    ref_image_idxs: list[int] | None = None
    ref_image_embeds: torch.Tensor | None = None
    ref_image_idx_mask: torch.BoolTensor | None = None

    """
    Basic string representation and property
    """

    def __str__(self):
        repr = []
        adjacency = (
            self.labeled_adjacency
            if self.labeled_adjacency is not None
            else self.adjacency
        )
        for prompt, bbox, adjacency in zip(self.prompts, self.bboxes, adjacency):
            bbox_str = ",".join([f"{x:.2f}" for x in bbox])
            repr.append(f"- {prompt} [{bbox_str}] [target: {adjacency}]")
        return "\n".join(repr)

    def __len__(self):
        return len(self.prompts)

    def get_layer_prompt(self, layer: int) -> str:
        bbox = self.bboxes[layer]
        bbox_str = ",".join([f"{x:.2f}" for x in bbox])
        return (
            f"Layer {layer}: {self.prompts[layer]} [{bbox_str}] "
            f"[target: {self.adjacency[layer]}]"
        )

    @property
    def roots(self) -> list[int]:
        is_root = [True] * len(self.adjacency)
        for adj in self.adjacency:
            for child in adj:
                is_root[child] = False
        return np.nonzero(is_root)[0].tolist()

    @property
    def topological_order(self) -> list[int]:
        dfs_order = []
        dfs_end_order = []
        visited = set()
        current_vertices = [(vertex_id, False) for vertex_id in self.roots]

        # Note that a vertex can be added multiple times if it is not a tree
        while len(current_vertices) > 0:
            # Using a stack
            vertex_id, processed = current_vertices.pop()
            if processed:
                dfs_end_order.append(vertex_id)
                continue
            if vertex_id in visited:
                continue
            visited.add(vertex_id)
            dfs_order.append(vertex_id)
            # record entering point
            current_vertices.append((vertex_id, True))
            # Reversing edge order here to get the correct order
            for child in reversed(self.adjacency[vertex_id]):
                if child not in visited:
                    current_vertices.append((child, False))

        return list(reversed(dfs_end_order))

    def set_ref_images(self, ref_images, ref_image_idxs: list[int]):
        self.ref_images = ref_images
        self.ref_image_idxs = ref_image_idxs
        self.ref_image_embeds = None
        self.ref_image_tensor_dict = None

    def add_ref_images(self, ref_images, ref_image_idxs: list[int]):
        if self.ref_images is None:
            self.ref_images = []
            self.ref_image_idxs = []
        if isinstance(self.ref_images, torch.Tensor):
            self.ref_images = list(torch.unbind(self.ref_images, dim=0))
        self.ref_images.extend(ref_images)
        self.ref_image_idxs.extend(ref_image_idxs)
        # For simplicity, reset cache ref
        self.ref_image_embeds = None
        self.ref_image_tensor_dict = None

    """
    Prepare for sampling
    """

    def convert_bboxes(self, reference_param: torch.Tensor):
        if not isinstance(self.bboxes, torch.Tensor):
            self.bboxes = torch.tensor(self.bboxes)
        self.bboxes = self.bboxes.to(reference_param)

    def convert_adjacency(self, tokenizer: PreTrainedTokenizerBase):
        if (
            self.converted_labeled_adjacency is None
            and self.labeled_adjacency is not None
        ):
            self.converted_labeled_adjacency = (
                GraphAttnMetaPerImage.from_prompts_and_adjacency(
                    self.prompts, self.labeled_adjacency, tokenizer
                )
            )
        return self.converted_labeled_adjacency

    def prepare_image_tensors(self, target_size: tuple[int, int]):
        if self.ref_image_tensor_dict is None:
            self.ref_image_tensor_dict = {}
        if target_size in self.ref_image_tensor_dict:
            return self.ref_image_tensor_dict[target_size]
        if isinstance(self.ref_images, ListConfig):
            self.ref_images = list(self.ref_images)
        ref_image_tensors = []
        for i, ref_image in enumerate(self.ref_images):
            if isinstance(ref_image, str):
                ref_image = Image.open(ref_image).convert("RGB")
                self.ref_images[i] = ref_image
            if isinstance(ref_image, Image.Image):
                ref_image = ref_image.resize(target_size, Image.LANCZOS)
                ref_image = T.ToTensor()(ref_image)
            assert isinstance(ref_image, torch.Tensor) and ref_image.size() == (
                3,
                target_size[0],
                target_size[1],
            )
            ref_image_tensors.append(ref_image)
        self.ref_image_tensor_dict[target_size] = torch.stack(ref_image_tensors)
        return self.ref_image_tensor_dict[target_size]

    def prepare_image_embeds(self, image_encoder: CLIPVisionModelWithProjection):
        if self.ref_image_embeds is not None or self.ref_images is None:
            return self.ref_image_embeds
        clip_image_size = image_encoder.config.image_size
        image_processor = CLIPImageProcessor(
            size=clip_image_size, crop_size=clip_image_size, do_rescale=False
        )
        ref_images = self.prepare_image_tensors((clip_image_size, clip_image_size))
        reference_param = next(image_encoder.parameters())
        ref_images = image_processor(ref_images, return_tensors="pt").pixel_values
        ref_images = ref_images.to(reference_param)
        with torch.no_grad():
            ref_image_embeds = image_encoder(ref_images).image_embeds
            uncond_image_embeds_single = torch.zeros_like(ref_image_embeds[0][None, :])
        # [b, 1280]
        uncond_image_embeds = uncond_image_embeds_single.tile([len(self.prompts), 1])
        cond_image_embeds = uncond_image_embeds.clone()
        ref_image_idx_mask = torch.zeros(
            len(self.prompts), dtype=torch.bool, device=reference_param.device
        )
        vertex_id_to_ref_image_pos_mapping = {
            vertex_id: i for i, vertex_id in enumerate(self.ref_image_idxs)
        }
        for i in range(len(self.prompts)):
            if i in vertex_id_to_ref_image_pos_mapping:
                cond_image_embeds[i] = ref_image_embeds[
                    vertex_id_to_ref_image_pos_mapping[i]
                ]
                ref_image_idx_mask[i] = True
        self.ref_image_embeds = cond_image_embeds
        self.ref_image_idx_mask = ref_image_idx_mask
        return self.ref_image_embeds

    """
    Image annotation
    """

    def get_labeled_bboxes(
        self, include_unlabeled: bool = False
    ) -> list[tuple[str, Bbox]]:
        labeled_bboxes = []
        vertices_to_process = set(range(len(self.prompts)))
        if self.labeled_adjacency is not None:
            for labeled_children in self.labeled_adjacency:
                for child, labels in labeled_children:
                    if child is None or child not in vertices_to_process:
                        continue
                    label = ", ".join(labels)
                    left, top, right, bottom = self.bboxes[child]
                    bbox = Bbox(left=left, top=top, right=right, bottom=bottom)
                    labeled_bboxes.append((label, bbox))
                    vertices_to_process.remove(child)
        if include_unlabeled:
            # Vertices without parents
            for vertex_id in vertices_to_process:
                left, top, right, bottom = self.bboxes[vertex_id]
                bbox = Bbox(left=left, top=top, right=right, bottom=bottom)
                labeled_bboxes.append(("", bbox))
        return labeled_bboxes

    def add_bbox_to_image(self, image: Image.Image):
        img_array = np.array(image)
        labeled_bboxes = self.get_labeled_bboxes()
        img_array_with_bboxes = annotate_all_labels(
            image=img_array, labeled_bboxes=labeled_bboxes
        )
        return Image.fromarray(img_array_with_bboxes)

    """
    Class instance construction
    """

    @classmethod
    def from_gbc_graph(
        cls, gbc_graph: GbcGraph, parent_prompt_must_include_text: bool = False
    ):
        prompts = []
        bboxes = []
        offset = 0
        vertex_id_to_prompt_indices = dict()
        for vertex in gbc_graph.vertices:
            prompts.extend([caption.text for caption in vertex.descs])
            bboxes.extend([vertex.bbox.to_xyxy() for _ in range(len(vertex.descs))])
            vertex_id_to_prompt_indices[vertex.vertex_id] = list(
                range(offset, offset + len(vertex.descs))
            )
            offset += len(vertex.descs)
        # Record adjacency with vertex id
        adjacency_tmp = [set() for _ in range(len(prompts))]
        labeled_adjacency_tmp = [dict() for _ in range(len(prompts))]
        for vertex in gbc_graph.vertices:
            for edge in vertex.out_edges:
                for prompt_idx in vertex_id_to_prompt_indices[vertex.vertex_id]:
                    if parent_prompt_must_include_text:
                        # Do not use edge if the corresponding prompt is not in the text
                        pattern = rf"\b{re.escape(edge.text)}\b"
                        if not bool(
                            re.search(pattern, prompts[prompt_idx], re.IGNORECASE)
                        ):  # noqa
                            continue
                    adjacency_tmp[prompt_idx].add(edge.target)
                    if edge.target not in labeled_adjacency_tmp[prompt_idx]:
                        labeled_adjacency_tmp[prompt_idx][edge.target] = set()
                    if vertex.label == "composition":
                        edge_text = re.sub(r"\s*\d+$", "", edge.text).strip()
                    else:
                        edge_text = edge.text
                    labeled_adjacency_tmp[prompt_idx][edge.target].add(edge_text)
        # Convert to prompt indices and the right format
        adjacency = []
        for adj in adjacency_tmp:
            prompt_adj = [vertex_id_to_prompt_indices[vertex_id] for vertex_id in adj]
            prompt_adj_flattened = [item for sublist in prompt_adj for item in sublist]
            adjacency.append(sorted(prompt_adj_flattened))
        labeled_adjacency = []
        for adj_dict in labeled_adjacency_tmp:
            labeled_prompt_adj = []
            for target, texts in adj_dict.items():
                target_prompts = vertex_id_to_prompt_indices[target]
                labeled_prompt_adj.extend(
                    [target_prompt, list(texts)] for target_prompt in target_prompts
                )
            labeled_adjacency.append(sorted(labeled_prompt_adj, key=lambda x: x[0]))
        return cls(
            prompts=prompts,
            bboxes=bboxes,
            adjacency=adjacency,
            labeled_adjacency=labeled_adjacency,
        )

    @classmethod
    def from_string(cls, text: str):
        return cls(
            prompts=[text],
            bboxes=[(0, 0, 1, 1)],
            adjacency=[[]],
            labeled_adjacency=[[]],
        )

    def new_gbc_prompt_from_str(
        self,
        prompts: str | list[str],
        copy_ref_images: bool = False,
        neg_prompt: bool = True,
    ) -> "GbcPrompt":
        "Create new gbc prompt from a string or a list of strings"
        if not isinstance(prompts, list):
            prompts = [prompts] * len(self.prompts)
        assert len(prompts) == len(self.prompts)
        new_gbc_prompt = GbcPrompt(
            prompts=prompts,
            bboxes=deepcopy(self.bboxes),
            adjacency=deepcopy(self.adjacency),
        )
        if neg_prompt and self.labeled_adjacency:
            for vertex_id, labeled_children in enumerate(self.labeled_adjacency):
                neg_labels = []
                for child, label in labeled_children:
                    if child is not None:
                        neg_labels.extend(label)
                        # neg_labels.extend(self.prompts[child])
                if neg_labels:
                    new_gbc_prompt.prompts[vertex_id] = (
                        ", ".join(neg_labels) + ", " + new_gbc_prompt.prompts[vertex_id]
                    )
        if copy_ref_images:
            new_gbc_prompt.ref_images = deepcopy(self.ref_images)
        return new_gbc_prompt

    def get_subgraph(self, vertex_ids: list[int]) -> "GbcPrompt":
        prompts = []
        bboxes = []
        bbox_is_tensor = isinstance(self.bboxes, torch.Tensor)
        adjacency = []
        labeled_adjacency = [] if self.labeled_adjacency is not None else None
        ref_image_embeds = [] if self.ref_image_embeds is not None else None
        ref_image_idx_mask = [] if self.ref_image_idx_mask is not None else None

        old_to_new_mapping = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}

        if self.ref_image_idxs is not None:
            assert self.ref_images is not None
            ref_image_idxs = []
            ref_images = []
            for i, ref_image in zip(self.ref_image_idxs, self.ref_images):
                if i in old_to_new_mapping:
                    ref_image_idxs.append(old_to_new_mapping[i])
                    ref_images.append(ref_image)
        else:
            ref_image_idxs = None
            ref_images = None

        for i, vertex_id in enumerate(vertex_ids):
            prompts.append(self.prompts[vertex_id])
            bboxes.append(self.bboxes[vertex_id])
            converted_children = []
            for child in self.adjacency[vertex_id]:
                if child in old_to_new_mapping:
                    converted_children.append(old_to_new_mapping[child])
            adjacency.append(converted_children)
            if self.labeled_adjacency is not None:
                converted_labeled_children = []
                for child, parent_texts in self.labeled_adjacency[vertex_id]:
                    # for caption mask
                    if child is None:
                        converted_labeled_children.append((None, parent_texts))
                    elif child in old_to_new_mapping:
                        converted_labeled_children.append(
                            (old_to_new_mapping[child], parent_texts)
                        )
                labeled_adjacency.append(converted_labeled_children)
            if ref_image_embeds is not None:
                ref_image_embeds.append(self.ref_image_embeds[vertex_id])
            if ref_image_idx_mask is not None:
                ref_image_idx_mask.append(self.ref_image_idx_mask[vertex_id])

        if ref_image_embeds is not None:
            ref_image_embeds = torch.stack(ref_image_embeds)
        if ref_image_idx_mask is not None:
            ref_image_idx_mask = torch.stack(ref_image_idx_mask)
        if bbox_is_tensor:
            bboxes = torch.stack(bboxes)

        subgraph = GbcPrompt(
            prompts=prompts,
            bboxes=bboxes,
            adjacency=adjacency,
            labeled_adjacency=labeled_adjacency,
            ref_image_idxs=ref_image_idxs,
            ref_images=ref_images,
            ref_image_embeds=ref_image_embeds,
            ref_image_idx_mask=ref_image_idx_mask,
        )
        return subgraph

    def sort_with_topological_order(self):
        return self.get_subgraph(self.topological_order)

    """
    For graph that concats parent prompts to child prompt
    """

    def get_depth_to_indices(self):
        depth_to_indices = []
        included = [True] * len(self.adjacency)
        for i in range(len(self.adjacency)):
            for child in self.adjacency[i]:
                included[child] = False
        nodes_current_level = np.nonzero(included)[0].tolist()
        depth_to_indices.append(sorted(nodes_current_level))
        while not np.all(included):
            nodes_next_level = []
            for node in nodes_current_level:
                for child in self.adjacency[node]:
                    if not included[child]:
                        nodes_next_level.append(child)
                        included[child] = True
            nodes_current_level = nodes_next_level
            depth_to_indices.append(sorted(nodes_current_level))
        return depth_to_indices

    def concat_ancestor_prompts(
        self,
        mask_out_concat: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
        margin: int = 5,
    ):
        if mask_out_concat:
            assert self.labeled_adjacency is not None
        depth_to_indices = self.get_depth_to_indices()
        graph_clone = deepcopy(self)
        graph_clone.converted_labeled_adjacency = None
        for depth in range(len(depth_to_indices) - 1):
            for parent in depth_to_indices[depth]:
                for child in self.adjacency[parent]:
                    child_prompt = graph_clone.prompts[child]
                    parent_prompt = self.prompts[parent]
                    if tokenizer is not None:
                        # Tokenize prompts
                        child_tokens = tokenizer.encode(
                            child_prompt, add_special_tokens=False
                        )
                        parent_tokens = tokenizer.encode(
                            parent_prompt, add_special_tokens=False
                        )

                        # Calculate total length
                        total_length = len(child_tokens) + len(parent_tokens) - 2

                        max_allowed_length = tokenizer.model_max_length - margin

                        if total_length > max_allowed_length:
                            # Number of tokens we can use from parent_prompt
                            available_length = max_allowed_length - len(child_tokens)

                            if available_length > 0:
                                parent_tokens = parent_tokens[1 : available_length + 1]
                                # Decode truncated tokens back to text
                                truncated_parent_prompt = tokenizer.decode(
                                    parent_tokens, skip_special_tokens=True
                                )
                            else:
                                # Not enough space for parent_prompt
                                # use child_prompt only
                                truncated_parent_prompt = ""
                        else:
                            # No truncation needed
                            truncated_parent_prompt = parent_prompt
                    # Update the child's prompt
                    if truncated_parent_prompt:
                        graph_clone.prompts[child] = (
                            f"{child_prompt} {truncated_parent_prompt}"
                        )
                    else:
                        graph_clone.prompts[child] = child_prompt
                    if mask_out_concat and truncated_parent_prompt:
                        graph_clone.labeled_adjacency[child].append(
                            (None, [truncated_parent_prompt])
                        )

        return graph_clone


if __name__ == "__main__":

    gbc_graphs = load_list_from_file(
        "data/gbc/prompt_gen/library_turtle_frog_steamponk.parquet", GbcGraph
    )
    print(gbc_graphs[2])
    print(GbcPrompt.from_gbc_graph(gbc_graphs[0]))
    print(GbcPrompt.from_gbc_graph(gbc_graphs[0]).get_labeled_bboxes())

    prompts = [
        "A cozy living room with a large sofa and a painting hanging on the wall above it.",  # noqa
        "The sofa is a plush, deep blue with soft cushions and a textured throw draped over one side.",  # noqa
        "The painting depicts a human riding a horse in a field.",
        "The living room has warm, ambient lighting from a nearby floor lamp.",
        "The person is wearing a red jacket and a blue beret.",
        "The field is filled with tall, golden grass swaying gently in the breeze.",
    ]
    bboxes = [[0, 0, 1, 1]] * 6
    adjacency = [[1, 2, 3], [], [4, 5], [], [], []]
    graph = GbcPrompt(prompts, bboxes, adjacency)
    print(graph.get_depth_to_indices())
