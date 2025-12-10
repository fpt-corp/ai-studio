mkdir -p /root/model
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B --local-dir=/root/model/Qwen/Qwen2.5-32B --local-dir-use-symlinks False