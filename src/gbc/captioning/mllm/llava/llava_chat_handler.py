# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

"""
Reference:
https://github.com/abetlen/llama-cpp-python

Conversation template:
https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py

Discussion
https://github.com/ggerganov/llama.cpp/pull/5267
https://github.com/ggerganov/llama.cpp/tree/4e96a812b3ce7322a29a3008db2ed73d9087b176/examples/llava

To adjust `model.config.image_grid_pinpoints` for fewer image tokens,
modify config_vit.json and convert again the mmproj model.
"""

import ctypes
import json
from typing import Optional, Union, Iterator

from llama_cpp import llama_types, llama, llama_grammar
from llama_cpp._utils import suppress_stdout_stderr
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    _get_system_message,
    _convert_completion_to_chat,
)


class Llava16ChatHandler(Llava15ChatHandler):

    def embed_image(self, image_bytes, llama_model):

        import array

        data_array = array.array("B", image_bytes)
        c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)
        with suppress_stdout_stderr(disable=self.verbose):
            embed = self._llava_cpp.llava_image_embed_make_with_bytes(
                self.clip_ctx,
                llama_model.context_params.n_threads,
                c_ubyte_ptr,
                len(image_bytes),
            )
        try:
            n_past = ctypes.c_int(llama_model.n_tokens)
            n_past_p = ctypes.pointer(n_past)
            with suppress_stdout_stderr(disable=self.verbose):
                self._llava_cpp.llava_eval_image_embed(
                    llama_model.ctx,
                    embed,
                    llama_model.n_batch,
                    n_past_p,
                )
            assert llama_model.n_ctx() >= n_past.value
            llama_model.n_tokens = n_past.value
        finally:
            with suppress_stdout_stderr(disable=self.verbose):
                self._llava_cpp.llava_image_embed_free(embed)


class LlavaYi34BChatHandler(Llava16ChatHandler):

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: list[llama_types.ChatCompletionRequestMessage],
        functions: Optional[list[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[list[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, list[str]]] = [],
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        assert (
            llama.context_params.logits_all is True
        )  # BUG: logits_all=True is required for llava
        assert self.clip_ctx is not None
        system_prompt = _get_system_message(messages)
        system_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>"
            if system_prompt != ""
            else "<|im_start|>system\nAnswer the questions.<|im_end|>"
        )
        user_role = "<|im_start|>user\n"
        assistant_role = "<|im_end|><|im_start|>assistant\n"
        llama.reset()
        llama.eval(llama.tokenize(system_prompt.encode("utf8"), add_bos=True))
        result = system_prompt
        for message in messages:
            if message["role"] == "user" and message["content"] is not None:
                if isinstance(message["content"], str):
                    llama.eval(
                        llama.tokenize(
                            f"{user_role} {message['content']}".encode("utf8"),
                            add_bos=False,
                        )
                    )
                    result += f"{user_role} {message['content']}"
                else:
                    assert isinstance(message["content"], list)
                    llama.eval(
                        llama.tokenize(f"{user_role}".encode("utf8"), add_bos=False)
                    )
                    result += f"{user_role}"
                    for content in message["content"]:
                        if content["type"] == "text":
                            llama.eval(
                                llama.tokenize(
                                    f"\n{content['text']}".encode("utf8"), add_bos=False
                                )
                            )
                            result += f"\n{content['text']}"
                        if content["type"] == "image":
                            result += "<image>"
                            image_bytes = content["image"]
                            self.embed_image(image_bytes, llama)
            if message["role"] == "assistant" and message["content"] is not None:
                llama.eval(
                    llama.tokenize(
                        (
                            "<|im_end|><|im_start|>assistant\n"
                            f"{message['content']}<|im_end|>"
                        ).encode("utf8"),
                        add_bos=False,
                    )
                )
                result += (
                    f"<|im_end|><|im_start|>assistant\n{message['content']}<|im_end|>"
                )
                assert llama.n_ctx() >= llama.n_tokens
        result += f"{assistant_role}"
        llama.eval(llama.tokenize(f"{assistant_role}".encode("utf8"), add_bos=False))
        assert llama.n_ctx() >= llama.n_tokens

        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            try:
                # create grammar from json schema
                if "schema" in response_format:
                    grammar = llama_grammar.LlamaGrammar.from_json_schema(
                        json.dumps(response_format["schema"])
                    )
            except Exception:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
            stream=stream,
        )


class LlavaMistral7BChatHandler(Llava16ChatHandler):

    def encode_user_text(self, text, llama_model, add_bos):
        llama_model.eval(llama_model.tokenize(text.encode("utf8"), add_bos=add_bos))

    def encode_user_content(self, content, llama_model, system_prompt=None):
        result = ""
        wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
        text = "[INST] "
        result += text
        self.encode_user_text(text, llama_model, (system_prompt is not None))

        if system_prompt is not None:
            result += wrap_sys(system_prompt)
            self.encode_user_text(system_prompt, llama_model, False)

        if isinstance(content, str):
            result += content
            self.encode_user_text(content, llama_model, False)
        else:
            assert isinstance(content, list)
            for content_item in content:
                if content_item["type"] == "text":
                    result += content_item["text"]
                    self.encode_user_text(content_item["text"], llama_model, False)
                if content_item["type"] == "image":
                    result += "<image>"
                    image_bytes = content_item["image"]
                    self.embed_image(image_bytes, llama_model)
        text = " [/INST]"
        result += text
        self.encode_user_text(text, llama_model, False)
        return result

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: list[llama_types.ChatCompletionRequestMessage],
        functions: Optional[list[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[list[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, list[str]]] = [],
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        assert (
            llama.context_params.logits_all is True
        )  # BUG: logits_all=True is required for llava
        assert self.clip_ctx is not None
        system_prompt = _get_system_message(messages)
        llama.reset()
        result = ""
        # The first message is system
        for i, message in enumerate(messages[1:]):
            if i == 0:
                assert (
                    message["role"] == "user"
                ), f"first message should come from user, get {message['role']}"
                message_content = message["content"]
                assert message_content, "first message should not be none"
                result += self.encode_user_content(
                    message_content, llama, system_prompt
                )
            elif i % 2 == 0:
                assert message["role"] == "user"
                result += self.encode_user_content(
                    message_content, llama, system_prompt=None
                )
            else:
                assert message["role"] == "assistant"
                message_content = " " + message["content"] + " </s>"
                result += message_content
                llama.eval(
                    llama.tokenize(message_content.encode("utf8"), add_bos=False)
                )
        # No need to add things for mistral instruct
        # result += ""
        # print(result)
        assert llama.n_ctx() >= llama.n_tokens

        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            try:
                # create grammar from json schema
                if "schema" in response_format:
                    grammar = llama_grammar.LlamaGrammar.from_json_schema(
                        json.dumps(response_format["schema"])
                    )
            except Exception:
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF
                )

        return _convert_completion_to_chat(
            llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
            ),
            stream=stream,
        )
