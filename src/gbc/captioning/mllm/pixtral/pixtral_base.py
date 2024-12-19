# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from functools import cache
from typing import Optional, Union
from PIL import Image

from vllm import LLM, SamplingParams

from gbc.utils import get_gbc_logger


@cache
def load_pixtral_model(
    model_name: str = "nm-testing/pixtral-12b-FP8-dynamic",
    max_num_seqs: int = 1,
    enforce_eager: bool = True,
    max_model_len: int = 8192,
    **kwargs,
):
    logger = get_gbc_logger()
    logger.info(f"Load Pixtral model from {model_name} ...")
    llm = LLM(
        model=model_name,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        **kwargs,
    )
    return llm


def pixtral_query_single(
    model: LLM,
    image: Image.Image,
    query: str,
    filled_in_query: Optional[Union[str, list[str], tuple[str]]] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
):
    if filled_in_query:
        if isinstance(filled_in_query, tuple) or isinstance(filled_in_query, list):
            query = query.format(*filled_in_query)
        else:
            query = query.format(filled_in_query)
    if system_message is not None:
        query = system_message + "\n" + query

    if verbose:
        logger = get_gbc_logger()
        logger.debug(f"Filled in query: {filled_in_query}")
        logger.debug(f"Query: {query}")

    inputs = {
        "prompt": f"<s>[INST]{query}.\n[IMG][/INST]",
        "multi_modal_data": {"image": [image]},
    }
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, **kwargs
    )
    outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    output = outputs[0].outputs[0].text

    return output
