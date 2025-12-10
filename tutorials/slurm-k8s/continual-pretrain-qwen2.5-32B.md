# Continual Pretraining of Qwen2.5-32B with Slurm on K8s

## Create environment

```bash
python3 –m venv venv 
source venv/bin/activate

# Setup LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel]" --no-build-isolation

# Install flash-attn
pip uninstall -y transformer-engine flash-attn && pip uninstall -y ninja && pip install ninja && pip -v install --no-cache-dir flash-attn --no-build-isolation
```
    
## Prepare Data
- Edit dataset
    ```bash
    vi data/dataset_info.json
    ```
- Append 2 lines to `dataset_info.json`
    ```json
    "culturay_vi": {"hf_hub_url": "htdung167/CulturaY_vi", "columns": {"prompt": "text"}},
    "culturay_vi_5gb": {"hf_hub_url": "htdung167/CulturaY_vi_5GB", "columns": {"prompt": "text"}},
    ```
    
## Download Qwen2.5-32B model
    
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B --local-dir=Qwen/Qwen2.5-32B --local-dir-use-symlinks False 
```
    
## LLaMA-Factory Training Config
- Create yaml file in examples folder
    ```bash
    vi examples/train.yaml
    ```
        
- Yaml content
        
    ```bash
    ### model
    model_name_or_path: ./Qwen2.5-32B
    trust_remote_code: true
    
    ### method
    stage: pt
    do_train: true
    finetuning_type: full
    deepspeed: examples/deepspeed/ds_z3_config.json
    
    ### dataset
    dataset: culturay_vi_5gb # culturay_vi if you want to training bigger dataset (35 GB)
    cutoff_len: 8192
    max_samples: 100000000000000000
    overwrite_cache: true
    preprocessing_num_workers: 16
    dataloader_num_workers: 4
    
    ### output
    output_dir: saves/pretrain_checkpoints
    logging_steps: 1
    save_steps: 100
    save_strategy: "no"
    plot_loss: true
    overwrite_output_dir: true
    save_only_model: false
    report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]
    
    ### train
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-4
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000
    resume_from_checkpoint: null
    flash_attn: "fa2"
    enable_liger_kernel: true
    
    ### eval
    # eval_dataset: c4_demo
    # val_size: 0.1
    # per_device_eval_batch_size: 1
    eval_strategy: "no"
    # eval_steps: 500
    ```
        
## Sbatch Config
- Create Sbatch file
    ```bash
    vi train_qwen.sbatch
    ```
- Sbatch content
    
    ```bash
    #!/bin/bash #SBATCH --job-name=multinode-training
    #SBATCH --nodes=4
    #SBATCH --time=2-00:00:00
    #SBATCH --gres=gpu:8
    #SBATCH -o training.out
    #SBATCH -e training.err
    #SBATCH --ntasks=4
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    node_id=${SLURM_NODEID}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | cut -d" " -f2)
    echo Master Node IP: $head_node_ip
    export LOGLEVEL=INFO
    export NNODES=4
    export NPROC_PER_NODE=8
    export MASTER_ADDR=$head_node_ip
    export MASTER_PORT=29500
    export NODE_RANK=$node_id
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=^lo,docker0
    export NCCL_TIMEOUT=180000000
    export NCCL_DEBUG=INFO
    export NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export FORCE_TORCHRUN=1   
    source venv/bin/activate
    srun llamafactory-cli train train.yaml
    ```
    
## Run Sbatch    
```bash
sbatch train_qwen.sbatch
```
    
## Check queue
```bash
squeue
```