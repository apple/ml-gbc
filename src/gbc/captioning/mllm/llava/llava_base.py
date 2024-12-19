# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import io
from typing import Optional, Literal, Union
from PIL import Image
from functools import cache

from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE

from gbc.utils import backoff, get_gbc_logger
from .llava_chat_handler import LlavaYi34BChatHandler, LlavaMistral7BChatHandler


def _image_to_bytes(image: Image.Image, format="JPEG"):
    # Convert the PIL Image to bytes
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        image_bytes = buffer.getvalue()
    return image_bytes


@cache
def load_llava_model(
    gpu_id=0,
    version: Literal["Yi-34B", "Mistral-7B"] = "Yi-34B",
    clip_model_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Llama:
    """
    Loads a LLaVA model using the
    `llama_cpp <https://github.com/abetlen/llama-cpp-python>`_
    library for efficient inference.

    Quantized gguf files can be found at
    https://huggingface.co/cmp-nct/llava-1.6-gguf/tree/main.

    As for documentation about making llava gguf file, please refer to
    https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/README.md.

    .. note::
       The loaded model is cached in memory so that repeated calls of the function
       with the same parameters would reuse the same model.

    Parameters
    ----------
    gpu_id
        The ID of the GPU to use.
    version
        The LLaVA model version to load.
    clip_model_path:
        Path to the vision tower of the LLaVA model. If not provided,
        the path will be retrieved from a predefined mapping based on the version.
    model_path
        Path to the LLM component of the LLaVA model. If not provided,
        the path will be retrieved from a predefined mapping based on the version.

    Returns
    -------
    Llama
        An instance of the loaded LLaVA model ready for querying.
    """

    logger = get_gbc_logger()

    path_mapping = {
        "Yi-34B": {
            "clip_model_path": "models/LLaVA-1.6/mmproj-llava-34b-f16-q6_k.gguf",
            "model_path": "models/LLaVA-1.6/ggml-yi-34b-f16-q_3_k.gguf",
        },
        "Mistral-7B": {
            "clip_model_path": "models/LLaVA-1.6/mmproj-mistral7b-f16-q6_k.gguf",
            "model_path": "models/LLaVA-1.6/ggml-mistral-q_4_k.gguf",
        },
    }

    if clip_model_path is None:
        clip_model_path = path_mapping[version]["clip_model_path"]
    if model_path is None:
        model_path = path_mapping[version]["model_path"]

    logger.info(
        f"Load Llava model {version} from {clip_model_path} and {model_path} ..."
    )

    if version == "Yi-34B":
        chat_handler = LlavaYi34BChatHandler(
            clip_model_path=clip_model_path,
            verbose=False,
        )
    else:
        chat_handler = LlavaMistral7BChatHandler(
            clip_model_path=clip_model_path,
            verbose=False,
        )
    model = Llama(
        model_path=model_path,
        n_gpu_layers=61,
        # n_ctx=2880 + 2048,  # depends on your system prompt and response length
        n_ctx=2880 + 3072,
        chat_handler=chat_handler,
        logits_all=True,
        verbose=False,
        split_mode=LLAMA_SPLIT_MODE_NONE,
        main_gpu=gpu_id,
    )

    return model


@backoff
def llava_query_single(
    model: Llama,
    image: Image.Image,
    query: str,
    filled_in_query: Optional[Union[str, list[str], tuple[str]]] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    # Both JPEG and PNG work but the results would be different
    image_format: str = "JPEG",
    verbose: bool = False,
    **kwargs,
):
    if filled_in_query:
        if isinstance(filled_in_query, tuple) or isinstance(filled_in_query, list):
            query = query.format(*filled_in_query)
        else:
            query = query.format(filled_in_query)
    if system_message is None:
        system_message = "Based on the content of image, answer user's question."

    if verbose:
        logger = get_gbc_logger()
        logger.debug(f"Filled in query: {filled_in_query}")
        logger.debug(f"System message: {system_message}")
        logger.debug(f"Query: {query}")

    result = model.create_chat_completion(
        temperature=temperature,
        max_tokens=max_tokens,
        # llama-cpp-python cannot detect the Eos token of Yi-34B correctly.
        # So we manually add the stop check.
        stop=["<|im"],
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": _image_to_bytes(image, format=image_format),
                    },
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ],
        **kwargs,
    )
    output = result["choices"][0]["message"]["content"]
    return output
