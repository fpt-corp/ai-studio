# Continual Pretraining of Qwen2.5-32B with Slurm on K8s

## Create environment (Login)
- Install miniconda3
    ```bash
    mkdir -p /root/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
    rm /root/miniconda3/miniconda.sh
    ```
- Create conda environment
    ```bash
    source /root/miniconda3/bin/activate
    conda create -n training python==3.10 -y
    conda activate training
    conda install pip
    pip install --upgrade pip

    # Setup LLaMA-Factory
    cd /root
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics,deepspeed,liger-kernel]" --no-build-isolation

    # Install flash-attn
    pip uninstall -y transformer-engine flash-attn && pip uninstall -y ninja && pip install ninja && pip -v install --no-cache-dir flash-attn --no-build-isolation

    # Install huggingface-cli, for downloading dataset and model
    pip install huggingface_hub
    ```
    
## Prepare Dataset
- Download dataset (Login)
    ```bash
    mkdir -p /root/data
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download htdung167/CulturaY_vi_5GB --local-dir /root/data/CulturaY_vi_5GB --local-dir-use-symlinks False --repo-type dataset
    ```
    
- Edit dataset info with the worker's path
    ```bash
    cd /root/LLaMA-Factory
    vi data/dataset_info.json
    ```
- Append this line to `dataset_info.json`
    ```json
    "culturay_vi_5gb": {"hf_hub_url": "/mnt/jail/root/data/CulturaY_vi_5GB", "columns": {"prompt": "text"}},
    ```
    
## Download Qwen2.5-32B model
    
```bash
mkdir -p /root/model
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B --local-dir=/root/model/Qwen/Qwen2.5-32B --local-dir-use-symlinks False 
```
    
## LLaMA-Factory Training Config
- Create yaml file in examples folder
    ```bash
    cd /root/LLaMA-Factory
    vi examples/train.yaml
    ```
        
- Yaml content
        
    ```bash
    ### model
    model_name_or_path: /mnt/jail/root/model/Qwen/Qwen2.5-32B
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
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 2.0e-5
    num_train_epochs: 2.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000
    resume_from_checkpoint: null
    flash_attn: "fa2"
    enable_liger_kernel: true

    ### eval
    eval_strategy: "no"
    ```
        
## Sbatch Config
- Create Sbatch file
    ```bash
    cd /root
    vi train_qwen.sbatch
    ```
- Sbatch content
    
    ```bash
    #!/bin/bash 
    #SBATCH --job-name=multinode-training
    #SBATCH --nodes=4
    #SBATCH --time=2-00:00:00
    #SBATCH --gres=gpu:8
    #SBATCH -o training_%j.out
    #SBATCH -e training_%j.err
    #SBATCH --ntasks=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}

    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | cut -d" " -f2)
    echo "Master Node: $head_node ($head_node_ip)"

    export LOGLEVEL=INFO
    export NNODES=$SLURM_NNODES # 4
    export NPROC_PER_NODE=8
    export MASTER_ADDR=$head_node_ip
    export MASTER_PORT=29500

    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=^lo,docker0
    export NCCL_DEBUG=INFO
    export NCCL_BLOCKING_WAIT=1
    export NCCL_ASYNC_ERROR_HANDLING=1

    export FORCE_TORCHRUN=1   

    srun bash -c '
        source /mnt/jail/root/miniconda3/bin/activate
        conda activate training
        
        echo "Node ID: $SLURM_NODEID - Hostname: $(hostname) - Master: $MASTER_ADDR"
        
        export NODE_RANK=$SLURM_NODEID
        
        cd /mnt/jail/root/LLaMA-Factory
        llamafactory-cli train examples/train.yaml
    '
    ```
    
## Run Sbatch    
```bash
sbatch train_qwen.sbatch
```
    
## Check queue
```bash
squeue
```
