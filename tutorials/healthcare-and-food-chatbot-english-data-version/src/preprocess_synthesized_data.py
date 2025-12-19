import os
import glob
import json
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import requests

def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_json(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

def load_student_prompts() -> Dict[str, str]:
    prompt_dir = "prompts/student_prompts"
    prompt_names = [
        "introduce_vn_food",
    ]

    return {name: read_txt(f"{prompt_dir}/{name}.txt") for name in prompt_names}

# Deep prompt
def process_deep_prompt():
    prompts = load_student_prompts()
    prompt_names = [
        "introduce_vn_food",
    ]
    for prompt_name in prompt_names:
        synthesized_data_dir = f"data/synthesized_data/{prompt_name}"
        save_dir = f"data/preprocessed_synthesized_data/{prompt_name}"
        os.makedirs(save_dir, exist_ok=True)

        synthesized_filepaths = glob.glob(f"{synthesized_data_dir}/*.json")

        for filepath in synthesized_filepaths:
            content = read_json(file_path=filepath)
            save_json_path = os.path.join(save_dir, os.path.basename(filepath))

            human_content = prompts[prompt_name].replace("[VN_FOOD]", content["food_content"])
            gpt_content = content["output"]

            save_content = {
                "conversations": [
                    {
                        "from": "human",
                        "value": human_content
                    },
                    {
                        "from": "gpt",
                        "value": gpt_content
                    }
                ]
            }
            write_json(file_path=save_json_path, content=save_content)

# Conversation prompt
def process_conversation_prompt():
    prompt_names = [
        "create_healthcare_conversation",
    ]
    for prompt_name in prompt_names:
        synthesized_data_dir = f"data/synthesized_data/{prompt_name}"
        save_dir = f"data/preprocessed_synthesized_data/{prompt_name}"
        os.makedirs(save_dir, exist_ok=True)

        synthesized_filepaths = glob.glob(f"{synthesized_data_dir}/*.json")

        for filepath in synthesized_filepaths:
            original_content = read_json(file_path=filepath)
            content = original_content["output"]
            save_json_path = os.path.join(save_dir, os.path.basename(filepath))

            start = content.find('[')
            end = content.rfind(']')

            if start == -1 or end == -1:
                print(content)
                continue

            json_str = content[start:end+1]

            try:
                unescaped = json.loads(json_str)
                if isinstance(unescaped, str):
                    records = json.loads(unescaped)
                else:
                    records = unescaped
            except json.JSONDecodeError as e:
                print(f"Error {filepath}: {e}")
                print(">>>:", json_str[:300])
                continue

            conversations = []
            for idx, item in enumerate(records):
                conversations.append({
                        "from": "human",
                        "value": item.get("user", "")
                    })
                conversations.append({
                    "from": "gpt",
                    "value": item.get("assistant", "")
                })

            output = {"conversations": conversations}
            write_json(file_path=save_json_path, content=output)

if __name__ == "__main__":
    process_deep_prompt()
    process_conversation_prompt()