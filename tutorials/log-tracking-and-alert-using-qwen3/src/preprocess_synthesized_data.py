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
        "json_analysis", 
    ]

    return {name: read_txt(f"{prompt_dir}/{name}.txt") for name in prompt_names}

# Json prompt
def process_json_prompt():
    def verify_format(actual, expected):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(verify_format(actual[k], expected[k]) for k in expected)
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                return False
            if expected:
                return all(isinstance(a, type(expected[0])) for a in actual)
            return True
        else:
            return isinstance(actual, type(expected))
    prompts = load_student_prompts()
    prompt_names = [
        "json_analysis",
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

            human_content = prompts[prompt_name].replace("[LOG_CONTENT]", original_content["log_content"])

            # Verify json
            start = content.find('{')
            end = content.rfind('}')
            json_part = content[start:end+1]
            try:
                unescaped = json.loads(json_part)
                if isinstance(unescaped, str):
                    data = json.loads(unescaped)
                else:
                    data = unescaped
            except json.JSONDecodeError as e:
                print(f"Error {filepath}: {e}")
                print(">>> ", json_part[:300])
                continue

            expected_format_data = {
                "risk_score": 100,
                "should_alert_user": True,
                "messages": {
                    "critical": [1, 5],
                    "warning": [7]
                },
                "summary": [
                    "Failed login attempts detected",
                    "Possible brute force attack"
                ]
            }
            if not verify_format(actual=data, expected=expected_format_data):
                print(data)
                continue

            save_content = {
                "conversations": [
                    {
                        "from": "human",
                        "value": human_content
                    },
                    {
                        "from": "gpt",
                        "value": json.dumps(data, ensure_ascii=False)
                    }
                ]
            }
            write_json(file_path=save_json_path, content=save_content)


if __name__ == "__main__":
    process_json_prompt()