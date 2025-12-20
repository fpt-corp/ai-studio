import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def count_tokens(tokenizer, message: list) -> int:
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

def analyze(tokenizer, data_file: str, split_name: str = "train", output_dir: str = "stats"):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_tokens_list = []
    for item in data:
        conv = item.get('conversations', [])
        new_conv = []
        for msg in conv:
            role = msg.get('from', '')
            content = msg.get('value', '')
            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
            new_conv.append({'role': role, 'content': content})
        num_tokens = count_tokens(tokenizer, new_conv)
        num_tokens_list.append(num_tokens)
    
    num_tokens_np = np.array(num_tokens_list)
    print(f"Dataset: {split_name}")
    print(f"Num samples: {len(num_tokens_np)}")
    print(f"Min tokens : {num_tokens_np.min()}")
    print(f"Max tokens : {num_tokens_np.max()}")
    print(f"Mean tokens: {num_tokens_np.mean():.2f}")
    print(f"Median     : {np.median(num_tokens_np)}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"num_tokens_distribution_{split_name}.png"
    )

    plt.figure(figsize=(10, 6))
    plt.hist(num_tokens_np, bins=40, edgecolor="black")
    plt.xlabel("Number of tokens")
    plt.ylabel("Number of samples")
    plt.title(f"Number of tokens distribution - {split_name}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = load_tokenizer(model_name)

    data_file = "data/final_data/healthcare_and_vn_food/test.json"
    analyze(tokenizer, data_file, split_name="test", output_dir="images")

    data_file = "data/final_data/healthcare_and_vn_food/train.json"
    analyze(tokenizer, data_file, split_name="train", output_dir="images")

    data_file = "data/final_data/healthcare_and_vn_food/val.json"
    analyze(tokenizer, data_file, split_name="val", output_dir="images")