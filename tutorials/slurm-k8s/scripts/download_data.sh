mkdir -p /root/data
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download htdung167/CulturaY_vi_5GB --local-dir /root/data/CulturaY_vi_5GB --local-dir-use-symlinks False --repo-type dataset