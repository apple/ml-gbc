# Install YOLO-World and get configs
git clone --recursive https://github.com/AILab-CVC/YOLO-World
cd YOLO-World
git checkout b449b98202e931590513c16e4830318be2dde946
python -m pip install .
cd ..

# https://github.com/open-mmlab/mmdetection/issues/12008
# python -m pip install torch==2.4.0 torchvision==0.19.0  ## Needed if no hacking_mmengine_history

# There is some issue in mmcv and mmdet version, so use mim for installation
# We get warning "yolo-world 0.1.0 requires mmdet==3.0.0, but you have mmdet 3.3.0 which is incompatible" but this should be fine
# see https://github.com/AILab-CVC/YOLO-World/issues/364 and https://github.com/AILab-CVC/YOLO-World/issues/279
python -m pip install -U openmim
mim install mmcv==2.1.0
mim install mmdet==3.3.0

python -m pip install huggingface_hub
python scripts/setup/download_yolo_world_models.py
