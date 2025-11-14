import random
import json
import os

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def reconstuct_data(data):
    reconstructed = []
    for item in data:
        new_value = f"Human: {item['instruction']}\nAssistant: {item['output']}\n"
        reconstructed.append({
            "text": new_value
        })
    return reconstructed

if __name__ == "__main__":
    path = "NLPLog.json"
    data = read_json(path)
    print(len(data))
    
    random.shuffle(data)

    split_ratio = 0.99
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_data = reconstuct_data(train_data)
    test_data = reconstuct_data(test_data)

    write_json("NLPLog_train.json", train_data)
    write_json("NLPLog_test.json", test_data)