## Setup

SAM segmentation model:

- pip install git+https://github.com/facebookresearch/segment-anything.git

SAM2 segmentation model:

- git clone https://github.com/facebookresearch/sam2
- cd sam2
- pip install -e .
- cd ..

Mask2Former segmentation model:

- pip install git+https://github.com/facebookresearch/detectron2.git
- pip install git+https://github.com/cocodataset/panopticapi.git
- git clone https://github.com/facebookresearch/Mask2Former
- cd mask2former/modeling/pixel_decoder/ops
- sh make.sh

Semantic-SAM segmentation model:

- git clone https://github.com/UX-Decoder/Semantic-SAM.git
- cd Semantic-SAM
- pip install -e .
- cd ..

CLIP (image to text model):

- pip install open_clip_torch

DinoV2 feature extractor model:

- pip install git+https://github.com/facebookresearch/dinov2.git

## (MSEG) GENERATING ROBUST SEGMENTATION LABEL MAPS

wsl
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3.10-venv python3.10-dev -y
cd mseg
python3.10 -m venv mseg_env_310
source mseg_env_310/bin/activate
git clone https://github.com/mseg-dataset/mseg-semantic.git
git clone https://github.com/mseg-dataset/mseg-api.git
cd mseg-semantic
pip install -e .
cd ..
cd mseg-api
pip install -e .
cd ..
cd ..
wget wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.run
sudo apt install build-essential
/usr/local/cuda-12.6/extras/demo_suite/deviceQuery
export PATH=/usr/local/cuda-12.6/bin:$PATH
nvidia-smi

pip install numpy==1.23.5

model_name=mseg-3m
model_path=./models/mseg-3m-480p.pth
input_file=./data/dataset_upscaled
config=./mseg/mseg-semantic/mseg_semantic/config/test/480/default_config_batched_ss.yaml
base_size=473

- python -u mseg/mseg-semantic/mseg_semantic/tool/universal_demo.py --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}

- input_file=./data/game_jpg

- python -u mseg/mseg-semantic/mseg_semantic/tool/universal_demo.py --config=${config} model_name ${model_name} model_path ${model_path} input_file ${input_file}

## Ais

- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/facebookresearch/sam2