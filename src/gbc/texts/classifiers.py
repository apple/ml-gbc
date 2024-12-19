# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import List, Tuple, Union
from functools import cache

import torch
import torch.nn as nn

import lightning.pytorch as pl
from sentence_transformers import SentenceTransformer

from gbc.utils import get_gbc_logger


@cache
def load_emb_model():
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
    )
    return model


@cache
def load_emb_classifier(model_path, gpu_id=0):
    logger = get_gbc_logger()
    logger.info(f"Loading text classifier from {model_path} on gpu {gpu_id}...")
    if torch.cuda.is_available():
        map_location = f"cuda:{gpu_id}"
    else:
        map_location = "cpu"
    model = EmbClassfier.load_from_checkpoint(
        model_path,
        emb_model=load_emb_model(),
        map_location=map_location,
    )
    model = model.eval()
    return model


@cache
def load_emb_pair_classifier(model_path, gpu_id=0):
    logger = get_gbc_logger()
    logger.info(f"Loading text pair classifier from {model_path} on gpu {gpu_id}...")
    if torch.cuda.is_available():
        map_location = f"cuda:{gpu_id}"
    else:
        map_location = "cpu"
    model = EmbPairClassfier.load_from_checkpoint(
        model_path,
        emb_model=load_emb_model(),
        map_location=map_location,
    )
    model = model.eval()
    return model


class EmbClassfier(pl.LightningModule):
    def __init__(
        self,
        emb_model: SentenceTransformer = None,
    ):
        assert emb_model is not None
        super(EmbClassfier, self).__init__()
        self.save_hyperparameters(ignore=["emb_model"])
        self.text_model = emb_model.eval().requires_grad_(False)
        test_output = self.text_model.encode(["hello", "world"])

        self.head = BinaryClassifierHead(test_output.shape[1])
        self.train_params = self.head.parameters()

    def forward(self, texts: str | List[str]):
        if isinstance(texts, str):
            texts = [texts]
        return self.head(self.text_model.encode(texts, convert_to_tensor=True))


class EmbPairClassfier(pl.LightningModule):
    def __init__(
        self,
        emb_model: SentenceTransformer = None,
    ):
        assert emb_model is not None
        super(EmbPairClassfier, self).__init__()
        self.save_hyperparameters(ignore=["emb_model"])
        self.text_model = emb_model.eval().requires_grad_(False)
        test_output = self.text_model.encode(["hello", "world"])

        self.head = PairClassifierHead(test_output.shape[1])
        self.train_params = self.head.parameters()

    def forward(self, text_pairs: Union[Tuple[str, str], Tuple[List[str], List[str]]]):
        texts1, texts2 = text_pairs
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]
        return self.head(
            self.text_model.encode(texts1, convert_to_tensor=True),
            self.text_model.encode(texts2, convert_to_tensor=True),
        )


class BinaryClassifierHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)


class PairClassifierHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim * 2),
            nn.Linear(in_dim * 2, in_dim * 8),
            nn.SiLU(),
            nn.Linear(in_dim * 8, 1),
        )

    def forward(self, x, y):
        return self.fc(torch.concat((x, y), dim=1))
