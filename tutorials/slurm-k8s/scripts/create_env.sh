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