# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from huggingface_hub import hf_hub_download


if __name__ == "__main__":

    hf_hub_download(
        repo_id="wondervictor/YOLO-World",
        filename="yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth",
        local_dir="models/yolo_world/",
    )
    hf_hub_download(
        repo_id="wondervictor/YOLO-World",
        filename="yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth",
        local_dir="models/yolo_world/",
    )
