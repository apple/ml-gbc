# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Set, Optional, Type, Literal
from functools import cache
from pydantic import BaseModel


class Description(BaseModel):
    text: str
    label: Literal[
        "short",
        "relation",
        "composition",
        "detail",
        "original",
        "hardcode",
        "bagofwords",
    ]

    # For consistency with `Caption`
    @classmethod
    def from_text_and_labels(cls, text: str, desc_label: str, **kwargs):
        return cls(text=text, label=desc_label)


class CaptionStatistics(BaseModel):
    n_words: int
    n_tokens: int

    @classmethod
    def from_caption(cls, caption: str):
        return cls(
            n_words=len(caption.split()),
            n_tokens=compute_n_tokens(caption),
        )


class ClipScores(BaseModel):

    # Empty dictionary can cause problem when saving in parquet
    scores: Optional[dict[str, float]] = None
    # This is to indicate if any truncation happens in clip encoding
    # This only happens if one single sentence is still too long after splitting
    truncation: bool = False

    def model_post_init(self, context):
        if self.scores == {}:
            self.scores = None


class Caption(Description):
    """
    This severs as a sort of buffer
    """

    full_label: Optional[str] = None
    statistics: Optional[CaptionStatistics] = None
    clip_scores: ClipScores = ClipScores()
    toxicity_scores: Optional[dict[str, float]] = None

    def model_post_init(self, context):
        if self.full_label is None:
            self.full_label = self.label
        if self.toxicity_scores == {}:
            self.toxicity_scores = None
        return self

    @classmethod
    def from_text_and_labels(
        cls, text: str, desc_label: str, vertex_label: Optional[str] = None
    ):
        if vertex_label is not None:
            full_label = f"{desc_label}-{vertex_label}"
        else:
            full_label = desc_label
        label = full_label.split("-")[0]
        return cls(text=text, label=label, full_label=full_label)

    def get_full_label_from_vertex_label(self, vertex_label: str):
        return f"{self.label}-{vertex_label}"

    @classmethod
    def from_desc_and_vertex_label(cls, desc: Description, vertex_label: str):
        return cls(
            text=desc.text, label=desc.label, full_label=f"{desc.label}-{vertex_label}"
        )

    def get_statistics(self):
        if self.statistics is None:
            self.statistics = CaptionStatistics.from_caption(self.text)
        return self.statistics

    @property
    def vertex_label(self):
        return self.full_label.split("-")[1]


@cache
def get_tokenizer(tokenizer_name="openai/clip-vit-large-patch14-336"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def compute_n_tokens(caption: str, tokenizer_name="openai/clip-vit-large-patch14-336"):
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(caption))


def get_bag_of_words_captions(
    texts: list[str],
    caption_type: Type[Description] = Description,
    max_n_tokens: Optional[int] = 77,
    vertex_label: Optional[str] = None,
    splitter=", ",
) -> tuple[list[Description], Set[str]]:
    """
    Create new captions from a list of words while ensuring
    these captions have at most `max_n_tokens` tokens
    """
    if max_n_tokens is None:
        text = splitter.join(texts)
        caption = Caption.from_text_and_labels(
            text=text, desc_label="bagofwords", vertex_label=vertex_label
        )
        return [caption], set()
    seen = set()
    # Respect the original order
    result = []
    for item in texts:
        if item not in seen:
            seen.add(item)
            result.append(item)
    texts = result
    texts_remaining = texts.copy()
    captions = []
    should_remove_texts = set()
    while len(texts_remaining) > 0:
        texts_remaining_update = []
        while len(texts_remaining) > 0:
            text_candidate = splitter.join(texts_remaining)
            caption = Caption.from_text_and_labels(
                text=text_candidate,
                desc_label="bagofwords",
                vertex_label=vertex_label,
            )
            if caption.get_statistics().n_tokens <= max_n_tokens:
                captions.append(
                    caption_type.from_text_and_labels(
                        text_candidate, "bagofwords", vertex_label=vertex_label
                    )
                )
                texts_remaining = []
            # This means that even with one word, the caption is too long
            elif len(texts_remaining) == 1:
                should_remove_texts.add(texts_remaining[0])
                texts_remaining = []
            else:
                texts_remaining_update.append(texts_remaining.pop())
        texts_remaining = texts_remaining_update
        texts_remaining.reverse()
    return captions, should_remove_texts
