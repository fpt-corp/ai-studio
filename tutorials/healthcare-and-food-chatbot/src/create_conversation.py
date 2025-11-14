import os
import glob
import json
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s — %(message)s',
    datefmt='%H:%M:%S'
)

def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def load_prompts() -> Dict[str, str]:
    prompt_dir = "prompts/teacher_prompts"
    prompt_names = [
        "create_healthcare_conversation",
    ]

    return {name: read_txt(f"{prompt_dir}/{name}.txt") for name in prompt_names}

def load_chunked_logs() -> List[dict]:
    all_items = []
    files = glob.glob("data/synthesized_data/introduce_vn_food/*.json")
    files = files * (2000 // len(files) + 1)
    for idx, path in enumerate(files):
        content = load_json(path)
        all_items.append(
            {
                "id": idx + 1,
                "content": content['output'],
                "no_lines": content['no_lines'],
                "food_type": content['food_content']
            }
        )
    random.shuffle(all_items)
    return all_items

def get_unprocessed_items(all_items, prompt_name):
    synth_dir = f"data/synthesized_data/{prompt_name}"
    os.makedirs(synth_dir, exist_ok=True)
    existing_ids = {
        os.path.splitext(f)[0] for f in os.listdir(synth_dir) if f.endswith('.json')
    }
    return [it for it in all_items if str(it['id']) not in existing_ids]

def call_openai(prompt: str, max_retries=3, delay=2):
    from openai import OpenAI
    keys = os.getenv("OPENAI_API_KEYS", "").split(",")
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(
        base_url=base_url,
        api_key=random.choice(keys).strip()
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return resp.choices[0].message.content, 200, "gpt-4o-mini"
        except Exception as e:
            logging.warning(f"[Retry {attempt+1}] OpenAI error: {e}")
            time.sleep(delay)
    return "Error fetching response", 500, "gpt-4o-mini"

def synthesize_one(prompt_name, items, prompts):
    for item in items:
        prompt = prompts[prompt_name].replace("[INFO_VN_FOOD]", item['content'])
        output_path = f"data/synthesized_data/{prompt_name}/{item['id']}.json"
        response, code, model = call_openai(prompt)
        if code != 200:
            logging.error(f"{item['id']} failed ({code}): {response}")
            continue
        save_json({
            "id": item['id'],
            "food_type": item['food_type'],
            "food_content": item['content'],
            "no_lines": item['no_lines'],
            "prompt_name": prompt_name,
            "prompt": prompt,
            "model_name": model,
            "output": response
        }, output_path)
        logging.info(f"Saved {prompt_name}/{item['id']}.json")


def run_all():
    prompts = load_prompts()
    all_items = load_chunked_logs()
    for prompt_name in prompts:
        unprocessed = get_unprocessed_items(all_items, prompt_name)
        logging.info(f"{prompt_name}: {len(unprocessed)} items to process")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(synthesize_one, prompt_name, [item], prompts)
                for item in unprocessed
            ]
            for _ in as_completed(futures):
                pass
    logging.info("All prompts processed successfully.")


if __name__ == "__main__":
    run_all()