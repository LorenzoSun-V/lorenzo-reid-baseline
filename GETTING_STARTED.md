# Getting Started

## Prepare the enviroment

### Get the docker image

This image is to use Xilinx pruning and quantizing tools. This step can be ignored if only do ReID training locally.

Pull the docker image by private docker hub(https://storagex.feishu.cn/docx/doxcn8wA8hBDgHVo9aI5pPqImah is the document of setting private docker hub):

```bash
# login private docker hub and input your username and password by prompt
docker login 192.168.2.83:5000
# pull docker image
docker pull 192.168.2.83:5000/lorenzo-vitis-ai-gpu:v0.1.4
```

### Git clone the project

Use following bash command to git clone ReID project if you are the member of the project:

```bash
git clone http://192.168.2.83/Lorenzo/lorenzo-reid-baseline.git
```

### Docker or local enviroment

1. Use following bash command to enter the docker enviroment:

```bash
cp lorenzo-reid-baseline/docker_run_no_license.sh .
# This script will map the current folder to /workspace folder in docker.
./docker_run_no_license.sh 192.168.2.83:5000/lorenzo-vitis-ai-gpu:v0.1.4
conda activate vitis-ai-pytorch
cd lorenzo-reid-baseline
```

2. Or use local environment, you need 
- Linux with python ≥ 3.6
- Pytorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`
- gdown (for automatically downloading pre-train model)
- sklearn
- termcolor
- tabulate
- [faiss](https://github.com/facebookresearch/faiss) `pip install faiss-cpu`

To set up with conda locally, you can run:

```bash
conda create -n reid python=3.7
conda activate reid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r lorenzo-reid-baseline/requirements.txt
cd lorenzo-reid-baseline
```

## Prepare pretrained model

If you use backbone in ./playreid/modeling/backbones, it will automatically download the pre-train models.
But if your network is not connected, you can download pre-train models manually and put it in `~/.cache/torch/checkpoints`.

If you want to use other pre-train models, such as MoCo pre-train, you can download by yourself and set the pre-train model path in `configs/Base-bagtricks.yml`.

## Compile with cython to accelerate evalution

```bash
cd playreid/evaluation/rank_cylib; make all
```

## Training & Evaluation in Command Line

The script "tools/train.py" made to train all the configs provided in this project.
You may want to use it as a reference to write your own training script.

To train a model with "train.py", first setup the corresponding datasets:

```bash
mkdir datasets
cd datasets
mkdir Market1501
cd Market1501
# This is Market1501 with uniform format.
wget http://192.168.2.78/img/private-dataset/lorenzo/market1501/lorenzo_market1501_v0.zip
unzip market1501/lorenzo_market1501_v0.zip
```

Then use "pwd" to acquire current dataset path, and change "dataset_dir=/workspace/..." in "playreid/data/datasets/market1501.py" to "dataset_dir=current_dataset_path".

This is the table of dataset urls:

|  Dataset name   | URL  |
| :----: | :----: |
|  Market1501  | http://192.168.2.78/img/private-dataset/lorenzo/market1501/lorenzo_market1501_v0.zip  |
|  CUHK03_l  | http://192.168.2.78/img/private-dataset/lorenzo/cuhk03_l/lorenzo_cuhk03_l_v0.zip |
|  DukeMTMC  | http://192.168.2.78/img/private-dataset/lorenzo/dukemtmc/lorenzo_dukemtmc_v0.zip |
|  MSMT17    | http://192.168.2.78/img/private-dataset/lorenzo/msmt17/lorenzo_msmt17_v0.zip   |

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config-file ./configs/Market1501/bagtricks_R50.yml
```

The configs are made for GPU-0 training.

If you want to train model with 2 GPUs, you can run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py --config-file ./configs/Market1501/bagtricks_R50.yml --num-gpus 2
```

The configs are made for GPU-0,1 DDP training.

If you want to train model with multiple machines, you can run:

```
# machine 1
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

python3 tools/train.py --config-file ./configs/Market1501/bagtricks_R50.yml \
--num-gpus 4 --num-machines 2 --machine-rank 0 --dist-url tcp://ip:port 

# machine 2
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

python3 tools/train.py --config-file ./configs/Market1501/bagtricks_R50.yml \
--num-gpus 4 --num-machines 2 --machine-rank 1 --dist-url tcp://ip:port 
```

Make sure the dataset path and code are the same in different machines, and machines can communicate with each other. 

To evaluate a model's performance, use

```bash
python3 tools/train.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only \
MODEL.TEST_WEIGHTS /path/to/checkpoint_file 
```

If you don't specify MODEL.TEST_WEIGHTS, "train.py" will use the weight "model_best.pth" generated in OUTPUT_DIR which is specified by yaml file automatically.

## Xilinx Prune&Quantize

1. Iterative prune + Quantize [model flow: .pth(train) -> sparse_model&slim_model.pth(prune) -> .xmodel(quantize)]

The script "applications/xilinx/iterprune_quantize.py" made to iterative prune and quantize automatically, will generate a xmodel and a txt containing bn params of the model which have not been quantized.

You can run:

```bash
python3 applications/xilinx/iterprune_quantize.py --config-file ./configs/Market1501/bagtricks_R50.yml \
--sparsity-ratios 0.3 0.5 0.6
```

to do iterative pruning and quantizing with sparsity=0.3, 0.5, 0.6.

2. Quantize directly [model flow: .pth(train) -> .xmodel(quantize)]

The script "applications/xilinx/quantize.py" made to quantize original pth automatically, will generate a xmodel and a txt containing bn params of the model which have not been quantized.

You can run:

```bash
python3 applications/xilinx/quantize.py --config-file ./configs/Market1501/bagtricks_R50.yml
```

to quantize the model directly.

## Compile the xmodel

If use docker environment, run like:

```bash
vai_c_xir  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -x ./logs/market1501/bagtricks_R50/quantizing/sparsity_0.5/Baseline_int.xmodel -o /workspace/zcu104_TEST -n reid
```

can compile the xmodel on zcu104.


