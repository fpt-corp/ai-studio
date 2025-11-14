import os
import json
import random

# ref: https://zenodo.org/records/8275861
LOG_FILES = {
    "Apache": "data/raw_data/Apache/Apache_full.log",
    "BGL": "data/raw_data/BGL/BGL_full.log",
    "Hadoop": "data/raw_data/Hadoop/Hadoop_full.log",
    "HDFS": "data/raw_data/HDFS/HDFS_full.log",
    "HealthApp": "data/raw_data/HealthApp/HealthApp_full.log",
    "HPC": "data/raw_data/HPC/HPC_full.log",
    "Linux": "data/raw_data/Linux/Linux_full.log",
    "Mac": "data/raw_data/Mac/Mac_full.log",
    "OpenSSH": "data/raw_data/OpenSSH/OpenSSH_full.log",
    "OpenStack": "data/raw_data/OpenStack/OpenStack_full.log",
    "Proxifier": "data/raw_data/Proxifier/Proxifier_full.log",
    "Spark": "data/raw_data/Spark/Spark_full.log",
    "Thunderbird": "data/raw_data/Thunderbird/Thunderbird_full.log",
    "Zookeeper": "data/raw_data/Zookeeper/Zookeeper_full.log",
}

OUTPUT_DIR = "data/chunked_logs"
NUM_SAMPLES = 300 # (300 * 14 - 74 -59)* 6 total ~24402 samples
MIN_LINES = 50
MAX_LINES = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

for dataset_name, log_path in LOG_FILES.items():
    if not os.path.exists(log_path):
        print(f"[WARN] File not found: {log_path}")
        continue

    print(f"[INFO] Processing {dataset_name}...")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"  - Total lines: {total_lines}")

    samples = []
    idx = 0

    for i in range(NUM_SAMPLES):
        if idx >= total_lines:
            print(f"  - End of file reached after {i} samples.")
            break

        sample_len = random.randint(MIN_LINES, MAX_LINES)
        sample_lines = lines[idx: idx + sample_len]
        idx += sample_len

        sample_id = f"{dataset_name}_{i+1:03d}"
        sample_text = "".join(sample_lines).strip()

        samples.append({
            "id": sample_id,
            "content": sample_text,
            "no_lines": sample_len
        })

    output_path = os.path.join(OUTPUT_DIR, f"chunked_{dataset_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"  - Saved {len(samples)} samples to {output_path}\n")

print("Done splitting all log files!")