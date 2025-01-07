# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import traceback
from queue import Queue
from threading import Thread

import torch
import transformers
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    stream_output=False,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    model_kwargs={},
    generation_kwargs={},
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **generation_kwargs,
    )
    model_generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
        **model_kwargs,
    }

    if stream_output:

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault("stopping_criteria", transformers.StoppingCriteriaList())
            kwargs["stopping_criteria"].append(Stream(callback_func=callback))
            with torch.no_grad(), autocast_gen():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs, callback=None)

        with generate_with_streaming(**model_generate_params) as generator:
            for output in generator:
                decoded_output = tokenizer.decode(output)
                if output[-1] == tokenizer.eos_token_id:
                    break
                yield decoded_output
        return  # early return for stream_output

    with torch.no_grad(), autocast_gen():
        generation_output = model.generate(**model_generate_params)
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield output


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs or {}
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except Exception:
                traceback.print_exc()

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            self.thread.join()
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
