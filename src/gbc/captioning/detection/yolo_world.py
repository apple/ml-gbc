# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from typing import Optional
from functools import cache

import numpy as np
import torch

# See https://github.com/open-mmlab/mmdetection/issues/12008
import gbc.captioning.detection.hack_mmengine_registry  # noqa

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast

from gbc.utils import get_gbc_logger
from .detection import Detection


@cache
def load_yolo_world_runner(
    model_version: str = "x_v2",
    work_dir: str = "logs",
    cfg_file: Optional[str] = None,
    model_path: Optional[str] = None,
    verbose: bool = False,
):
    logger = get_gbc_logger()
    logger.info("Load YoloWorld model...")
    if cfg_file is None:
        if model_version == "l_v2":
            cfg_file = "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"  # noqa: E501
        elif model_version == "x_v2":
            cfg_file = "yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"  # noqa: E501
        else:
            raise ValueError("Unsupported YoloWorld model version")
        cfg_file = os.path.join("YOLO-World", "configs", "pretrain", cfg_file)
    if model_path is None:
        if model_version == "l_v2":
            model_name = "yolo_world_v2_l_obj365v1_goldg_cc3mv2_pretrain-2f3a4a22.pth"
        if model_version == "x_v2":
            model_name = "yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pth"
        model_path = os.path.join("models", "yolo_world", model_name)
    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = work_dir
    cfg.load_from = model_path
    if not verbose:
        cfg.log_level = "WARNING"
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    # Remove the load image part to process image directly
    pipeline = cfg.test_dataloader.dataset.pipeline[1:]
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    return runner


class YoloWorldDetection(Detection):
    """
    Detection wrapper for
    `YOLO-World <https://github.com/AILab-CVC/YOLO-World/>`_ model.

    .. note::
       The loaded model is cached in memory so that repeated instantiations of
       the class with the same parameters would reuse the same model.

    Attributes
    ----------
    model_version : str, default="x_v2"
        Model version.
    work_dir : str, default="logs"
        Directory to save the log files.
    cfg_file : str | None, default=None
        Path to the configuration file.
        When set to ``None``, it loads config file according to ``model_version``.
    model_path : str | None, default=None
        Path to the model file.
        When set to ``None``, it loads model file according to ``model_version``.
    verbose : bool, default=False
        Whether to print the log information.
    """

    def __init__(
        self,
        model_version: str = "x_v2",
        work_dir: str = "logs",
        cfg_file: Optional[str] = None,
        model_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.runner = load_yolo_world_runner(
            model_version=model_version,
            work_dir=work_dir,
            cfg_file=cfg_file,
            model_path=model_path,
            verbose=verbose,
        )

    def detect_core(
        self, image: np.ndarray, texts: list[str]
    ) -> tuple[list[tuple[int, int, int, int]], list[float], list[int]]:
        # Convert to BGR as yolo_world uses cv2 backend
        image = image[:, :, ::-1]

        _texts = [[text.strip()] for text in texts] + [[" "]]

        data_info = self.runner.pipeline(
            dict(img_id=0, img=image, texts=_texts, ori_shape=image.shape[:2])
        )

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        pred_instances = pred_instances.cpu()

        bboxes = pred_instances["bboxes"].tolist()
        scores = pred_instances["scores"].tolist()
        labels = pred_instances["labels"].tolist()

        bboxes_update, scores_update, labels_update = [], [], []
        for bbox, score, label in zip(bboxes, scores, labels):
            # For some reason, the label can be out of index
            # Seems to correspond to the empty string placeholder
            if label >= len(texts):
                logger = get_gbc_logger()
                logger.warning(f"Detected index out of range {label} for {texts}")
                continue
            bboxes_update.append(bbox)
            scores_update.append(score)
            labels_update.append(label)

        return bboxes_update, scores_update, labels_update
