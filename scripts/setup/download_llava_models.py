# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from huggingface_hub import hf_hub_download


if __name__ == "__main__":

    hf_hub_download(
        repo_id="cmp-nct/llava-1.6-gguf",
        filename="mmproj-mistral7b-f16-q6_k.gguf",
        local_dir="models/LLaVA-1.6/",
    )
    hf_hub_download(
        repo_id="cmp-nct/llava-1.6-gguf",
        filename="ggml-mistral-q_4_k.gguf",
        local_dir="models/LLaVA-1.6/",
    )
    hf_hub_download(
        repo_id="cmp-nct/llava-1.6-gguf",
        filename="mmproj-llava-34b-f16-q6_k.gguf",
        local_dir="models/LLaVA-1.6/",
    )
    hf_hub_download(
        repo_id="cmp-nct/llava-1.6-gguf",
        filename="ggml-yi-34b-f16-q_3_k.gguf",
        local_dir="models/LLaVA-1.6/",
    )
