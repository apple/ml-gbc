# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
The additional complexity in this file comes from the following two considerations
1. batch processing
2. Avoid repeated image encoding (multiple texts to encode for each cropped image)
"""

from typing import Optional, Callable
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import nltk
from nltk.tokenize import sent_tokenize
from transformers import CLIPProcessor, CLIPModel
from open_clip import create_model_from_pretrained, get_tokenizer

from gbc.utils import get_gbc_logger
from gbc.data.bbox import crop_bbox
from gbc.data.graph import GbcGraph, GbcGraphFull
from gbc.data.caption import Caption, get_bag_of_words_captions


def compute_clip_scores(
    gbc_graphs: list[GbcGraph],
    clip_models: dict[str, "ClipScoreModel"],
    batch_size: int = 512,
) -> list[GbcGraphFull]:
    """
    Compute CLIP scores for GBC graphs.

    This function calculates CLIP scores for all captions associated with vertices
    in the provided GBC graphs.
    Images are cropped based on vertex bounding boxes, and scores are computed
    in batches for efficiency.

    Parameters
    ----------
    gbc_graphs
        A list of GBC graphs for which the CLIP scores need to be computed.
        If a :class:`~GbcGraph` is passed,
        a new :class:`~GbcGraphFull` instance is created.
    clip_models
        A dictionary of CLIP models where keys are model names and values are model
        instances for computing CLIP scores.
    batch_size : int, optional
        The size of batches for processing captions and images, by default 512.

    Returns
    -------
    list of GbcGraphFull
        A list of :class:`~GbcGraphFull` objects with updated CLIP scores.

    Notes
    -----
    - If an input ``gbc_graph`` is already in :class:`~GbcGraphFull`,
      the modifications will be in-place.
    - Logs a warning and skips graphs where images are not found locally or via URL.
    """

    buffer = CaptionImageBuffer(batch_size=batch_size, clip_models=clip_models)
    logger = get_gbc_logger()
    gbc_graphs_with_clip_scores = []
    # This is to avoid repeated encoding of same image
    cropped_image_id = 0

    for gbc_graph in tqdm(gbc_graphs, desc="Computing clip scores"):
        if not isinstance(gbc_graph, GbcGraphFull):
            gbc_graph = GbcGraphFull.from_gbc_graph(gbc_graph)
        image = gbc_graph.get_image()
        if image is None:
            logger.warning(
                f"Image not found for graph with path {gbc_graph.img_path} "
                f"and url {gbc_graph.img_url}. Skipping..."
            )
            gbc_graphs_with_clip_scores.append(gbc_graph)
            continue
        image = np.array(image)
        for vertex in gbc_graph.vertices:
            cropped_image = Image.fromarray(crop_bbox(image, vertex.bbox))
            for caption in vertex.descs:
                buffer.add(
                    caption, cropped_image, str(cropped_image_id), flush_if_full=False
                )
            cropped_image_id += 1
        gbc_graphs_with_clip_scores.append(gbc_graph)

    buffer.flush_buffer(flush_all=True)
    return gbc_graphs_with_clip_scores


class TextImageIdPair(object):

    def __init__(self, text: str, image_id: str):
        self.text = text
        self.image_id = image_id
        self.clip_scores: Optional[dict[str, float]] = None


class ClipScoreModel(object):

    def __init__(self, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip_model.to(self.device)

    def compute_scores(
        self,
        text_image_id_pairs: list[TextImageIdPair],
        image_dict: dict[str, Image.Image],
    ) -> np.array:
        assert len(text_image_id_pairs) > 0, "Empty text_image_id_pairs"
        texts = [t.text for t in text_image_id_pairs]
        sorted_images = sorted(image_dict.items(), key=lambda x: x[0])
        image_ids, images = zip(*sorted_images)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            text_features = self.get_normalize_text_features(texts)
            image_features = self.get_normalize_image_features(images)
            # Find the corresponding image features for each text
            image_feature_dict = {
                image_id: image_feature
                for image_id, image_feature in zip(image_ids, image_features)
            }
            image_features = torch.stack(
                [image_feature_dict[t.image_id] for t in text_image_id_pairs]
            )
            scores = torch.sum(image_features * text_features, dim=-1)
        # from float32 to float
        return scores.cpu().numpy().astype(float)

    def get_normalize_text_features(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def get_normalize_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError


class HfClipScoreModel(ClipScoreModel):

    def __init__(self, clip_model_path, processor_model_path=None, device=None):
        print(f"Loading CLIP model from {clip_model_path}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        if processor_model_path is not None:
            self.processor = CLIPProcessor.from_pretrained(processor_model_path)
        else:
            self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        super().__init__(device)

    def get_normalize_text_features(self, texts: list[str]) -> torch.Tensor:
        # padding with 49407 (the index of [end of text])
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    def get_normalize_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)


class OpenClipScoreModel(ClipScoreModel):

    def __init__(self, clip_model_path, tokenizer_name="ViT-H-14", device=None):
        print(f"Loading CLIP model from {clip_model_path}")
        self.clip_model, self.preprocess = create_model_from_pretrained(clip_model_path)
        self.tokenizer = get_tokenizer(tokenizer_name)
        super().__init__(device)

    def get_normalize_text_features(self, texts: list[str]) -> torch.Tensor:
        # padding with 0 -> This does not really matter
        inputs = self.tokenizer(
            texts, context_length=self.clip_model.context_length
        ).to(self.device)
        text_features = self.clip_model.encode_text(inputs)
        return F.normalize(text_features, dim=-1)

    def get_normalize_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = torch.stack([self.preprocess(image) for image in images]).to(
            self.device
        )
        image_features = self.clip_model.encode_image(inputs)
        return F.normalize(image_features, dim=-1)


class TextImageBuffer(object):

    def __init__(self, batch_size: int, clip_models: dict[str, ClipScoreModel]):
        self.image_buffer: dict[str, Image.Image] = {}
        self.text_image_id_pairs: list[TextImageIdPair] = []
        self.batch_size = batch_size
        self.clip_models = clip_models

    def add(
        self,
        text_image_id_pair: TextImageIdPair,
        image: Image.Image = None,
        flush_if_full: bool = True,
    ) -> bool:
        if text_image_id_pair.image_id not in self.image_buffer and image:
            self.image_buffer[text_image_id_pair.image_id] = image
        self.text_image_id_pairs.append(text_image_id_pair)
        if len(self.text_image_id_pairs) >= self.batch_size and flush_if_full:
            self.flush_buffer()
            return True
        else:
            return False

    def flush_buffer(self):
        if len(self.text_image_id_pairs) == 0:
            return
        clip_scores = dict()
        for clip_name, clip_model in self.clip_models.items():
            clip_scores[clip_name] = clip_model.compute_scores(
                self.text_image_id_pairs, self.image_buffer
            )
        # Convert dict of list to list of dict
        clip_scores = [
            {k: v[i] for k, v in clip_scores.items()}
            for i in range(len(self.text_image_id_pairs))
        ]
        for text_image_id_pair, clip_score in zip(
            self.text_image_id_pairs, clip_scores
        ):
            text_image_id_pair.clip_scores = clip_score
        self.text_image_id_pairs = []
        # We flush the image buffer, so we may have repeated encoding if
        # the same cropped region is used in different batches
        self.image_buffer = {}


class CaptionImageBuffer(object):

    def __init__(
        self,
        batch_size: int,
        clip_models: dict[str, ClipScoreModel],
        agg_func: Callable = np.mean,
        recompute_clip_scores: bool = False,
    ):
        nltk.download("punkt_tab", quiet=True)
        self.caption_buffer = []
        self.text_image_buffer = TextImageBuffer(batch_size, clip_models)
        self.agg_func = agg_func
        self.clip_names = clip_models.keys()
        self.recompute_clip_scores = recompute_clip_scores

    def add(self, caption, image, image_id, flush_if_full=True):
        if caption.clip_scores.scores is None:
            caption.clip_scores.scores = {}
        if (
            np.all(
                [
                    clip_name in caption.clip_scores.scores
                    for clip_name in self.clip_names
                ]
            )
            and not self.recompute_clip_scores
        ):
            return False
        text_image_id_pairs = []
        buffer_flushed = False
        for text in self.split_caption(caption):
            text_image_id_pair = TextImageIdPair(text, image_id)
            text_image_id_pairs.append(text_image_id_pair)
            buffer_flushed |= self.text_image_buffer.add(
                text_image_id_pair, image=image, flush_if_full=flush_if_full
            )
        # Store the correspondence between caption and its text image id pairs
        self.caption_buffer.append((caption, text_image_id_pairs))
        if buffer_flushed:
            self.flush_buffer(flush_all=False)
            return True
        else:
            return False

    def flush_buffer(self, flush_all: bool = True):
        """
        Process caption_image_buffer to get (average) clip scores
        for each caption

        Parameters
        ----------
        flush_all (bool): whether to flush all the captions in the buffer
            If set to True, it will flush the text_image_buffer to get
            all the clip scores.
        """
        if flush_all:
            self.text_image_buffer.flush_buffer()
        remaining = []
        for caption, text_image_id_pairs in self.caption_buffer:
            clip_scores = {clip_name: [] for clip_name in self.clip_names}
            completed = True
            for text_image_id_pair in text_image_id_pairs:
                if text_image_id_pair.clip_scores is None:
                    completed = False
                    remaining.append((caption, text_image_id_pairs))
                    break
                else:
                    for clip_name in self.clip_names:
                        clip_scores[clip_name].append(
                            text_image_id_pair.clip_scores[clip_name]
                        )
            if completed:
                caption.clip_scores.scores.update(
                    {
                        clip_name: self.agg_func(clip_scores[clip_name])
                        for clip_name in self.clip_names
                    }
                )
        self.caption_buffer = remaining

    @staticmethod
    def split_caption(caption: Caption):
        n_tokens = caption.get_statistics().n_tokens
        if n_tokens > 77:
            sentences = sent_tokenize(caption.text)
            concats, longer_than_77 = get_bag_of_words_captions(
                sentences, max_n_tokens=77, splitter=" "
            )
            if len(longer_than_77) > 0:
                caption.clip_scores.truncation = True
            return [concat.text for concat in concats] + list(longer_than_77)
        else:
            return [caption.text]
