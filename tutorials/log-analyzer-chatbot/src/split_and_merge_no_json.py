import os
import json
import random
from pathlib import Path
random.seed(42)

BASE_DIR = Path("data/preprocessed_synthesized_data")
VAL_SAMPLES = 100
TEST_SAMPLES = 100
TRAIN_SAMPLES_SMALL = 1000
OUTPUT_DIR = Path("data/final_data/chat")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_json_files(folder):
    data = []
    for f in sorted(Path(folder).glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = json.load(file)
                data.append(content)
        except Exception as e:
            print(f"Error {f}: {e}")
    return data

train_all, val_all, test_all = [], [], []
train_all_small = []

for subfolder in BASE_DIR.iterdir():
    if "json_analysis" in subfolder.name:
        continue
    if subfolder.is_dir():
        all_data = load_json_files(subfolder)
        if not all_data:
            continue

        random.shuffle(all_data)

        val_data = all_data[:VAL_SAMPLES]
        test_data = all_data[VAL_SAMPLES:VAL_SAMPLES + TEST_SAMPLES]
        train_data = all_data[VAL_SAMPLES + TEST_SAMPLES:]
        train_data_small = train_data[:TRAIN_SAMPLES_SMALL]

        val_all.extend(val_data)
        test_all.extend(test_data)
        train_all.extend(train_data)

        train_all_small.extend(train_data_small)

        print(f"{subfolder.name}: total={len(all_data)}, "
              f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}, "
              f"train_small={len(train_data_small)}")

random.shuffle(train_all)
random.shuffle(val_all)
random.shuffle(test_all)
random.shuffle(train_all_small)

splits = {
    "train_no_json": train_all,
    "val_no_json": val_all,
    "test_no_json": test_all,
    "train_no_json_small": train_all_small
}

for name, data in splits.items():
    out_path = OUTPUT_DIR / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {name}: {len(data)} samples -> {out_path}")
