## Environment Setup
```bash
sudo -H env "PATH=$PATH" pip install -r requirements.txt
```

## pretrain
```bash
# [option 1] distributed: NOT available now
export TORCH_DISTRIBUTED_DEBUG=INFO # ddp detailed info
export DISABLE_MLFLOW_INTEGRATION=TRUE # shut down mlflow to avoid log error
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=2 --nnodes=4 --master_port=3192  pretrain.py \
    --micro_batch_size 1 \
    --data_percent 1


# [option 2] non-ddp
export DISABLE_MLFLOW_INTEGRATION=TRUE # shut down mlflow to avoid log error
python pretrain.py \
    --micro_batch_size 128 \
    --data_percent 1

# [option 3] debug
# Trick: we use `none` in bash scripts when we want to pass `None` in python
# data percent as 13~10/84 to align with our setting in full model pretrain
export DISABLE_MLFLOW_INTEGRATION=TRUE # shut down mlflow to avoid log error
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192  pretrain.py \
    --data_path yahma/alpaca-cleaned \
    --data_split none \
    --batch_size 2 \
    --cutoff_len 64 \
    --prompt_template_name alpaca \
    --micro_batch_size 1 \
    --data_percent 1 \
    --output_dir /tmp/test \
    --debug

export DISABLE_MLFLOW_INTEGRATION=TRUE # shut down mlflow to avoid log error
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192  finetune.py \
    --data_path yahma/alpaca-cleaned \
    --batch_size 2 \
    --cutoff_len 64 \
    --micro_batch_size 1 \
    --output_dir /tmp/test \
    --debug

python pretrain.py \
    --micro_batch_size 64 \
    --data_percent 1 \
    --output_dir /tmp/test 

```
### possible errors: 
* `ModuleNotFoundError: No module named 'torch._six'`: -change `torch._six` to `torch`
* `cannot import name 'string_classes' from 'torch'`: change the code to this
```python
# from torch import string_classes
string_classes = str
```
* wandb: 
solution1:
```bash
# choose option1 by typing
1
# past my wandb api key
93d5c46355069839af310dd152c66023971159bf
```

* `RuntimeError: expected scalar type Half but found Float" when using fp16`
add one line before the trainer.train in [pretrain.py](pretrain.py)
```python
with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```
solution 2: disable wandb by setting this in [pretrain.py](pretrain.py), seems failed, pls go back to solution1 😂
```python
wandb_watch: str = "false",  # options: false | gradients | all
wandb_log_model: str = "false",  # options: false | true
```


### Docker Setup & Inference
0. apt get docker

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install -y nvidia-container-toolkit
```

1. Build the container image:

```bash
sudo -H env "PATH=$PATH" dockerd
sudo -H env "PATH=$PATH"  docker build -t alpaca-lora .
```
NOTE: each time tou met errors, try restart docker by `ctrl+c` old docker service and then `sudo dockerd`

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
sudo -H env "PATH=$PATH" docker run --gpus=all --shm-size 64g -p 7860:7860 -v /mnt/msranlpintern/daixuan/cache --rm alpaca-lora pretrain.py \
    --micro_batch_size 8 \
    --data_percent 1 \
    --output_dir /tmp/test 
```

3. Open `https://localhost:7860` in the browser
