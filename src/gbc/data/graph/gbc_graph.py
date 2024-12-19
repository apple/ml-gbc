# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from PIL import Image
from typing import Literal, Optional
from pydantic import BaseModel

from gbc.utils import ImageCache
from ..bbox import Bbox
from ..caption import Description


class GbcEdge(BaseModel):
    # Source and targets are vertices id
    source: str
    text: str
    target: str


class GbcVertex(BaseModel):
    vertex_id: str
    bbox: Bbox
    label: Literal["image", "entity", "composition", "relation"]
    descs: list[Description]
    in_edges: list[GbcEdge] = []
    out_edges: list[GbcEdge] = []


class GbcGraph(BaseModel):

    vertices: list[GbcVertex]
    img_url: Optional[str] = None
    img_path: Optional[str] = None
    original_caption: Optional[str] = None
    short_caption: Optional[str] = None
    detail_caption: Optional[str] = None

    _image_cache: Optional[ImageCache] = None

    def model_post_init(self, context):
        for vertex in self.vertices:
            if vertex.label == "image":
                for desc in vertex.descs:
                    if desc.label == "original" and self.original_caption is None:
                        self.original_caption = desc.text
                    elif desc.label == "short" and self.short_caption is None:
                        self.short_caption = desc.text
                    elif desc.label == "detail" and self.detail_caption is None:
                        self.detail_caption = desc.text
                break

    def get_image(self, img_root_dir: str = "") -> Optional[Image.Image]:
        if self._image_cache is None:
            self._image_cache = ImageCache(
                img_path=os.path.join(img_root_dir, self.img_path),
                img_url=self.img_url,
            )
        return self._image_cache.get_image()
