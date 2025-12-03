# FPT Model Hub SDK CLI Command

## 1. Private Model
### 1.1 Log in with your API Token
Login with environment variables

```bash
export FPT_SPACE_URL=https://ai-api.fptcloud.com 
export FPT_TENANT_ID=YOUR-TENANT-ID
export FPT_SPACE_TOKEN=YOUR-PERSONAL-TOKEN
```
### 1.2 List model
```bash
model_space model ls
```
Example result:
```
2025-12-02 15:18:15,367 [INFO] Get models list from FPT Model Hub - Private Model
                                                                       FPT Model Hub - Private Model                                                                        
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    ┃                                   ID ┃                                               Name ┃          Description ┃ Number of versions ┃                  Updated At ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  0 │ 53cad226-3af0-4f55-b0a4-2ef98414ffec │                 qwen-lora-ft.finetune-ccc629cab21b │ Qwen2.5-32B-Instruct │                  1 │ 2025-12-02T07:50:10.599147Z │
└────┴──────────────────────────────────────┴────────────────────────────────────────────────────┴──────────────────────┴────────────────────┴─────────────────────────────┘
```
### 1.3 Create model
```python
model_space model create --model-name <model-name> 
```
Example result:
```
2025-12-02 16:08:43,876 [INFO] Creating new model...
2025-12-02 16:08:44,546 [INFO] Model is created
```
### 1.4 Delete model
```
model_space model delete --model-id <model-id> 
```
Example result:
```
Are you sure you want to delete model 'ad2c8ebf-e85d-408c-9540-b560e6e330ec'? [y/N]: y
2025-12-02 16:12:10,667 [INFO] Model ad2c8ebf-e85d-408c-9540-b560e6e330ec is deleted!
```
### 1.5 List version
```
model_space model version ls --model-id <model-id>
```
Example result:
```
2025-12-02 16:14:04,824 [INFO] Get model versions list from FPT Model Hub - Private Model
                                 FPT Model Hub - Private Model Version                                  
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   ┃                           Version ID ┃ Version Name ┃ Version Size ┃                  Updated At ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0 │ e9436dff-2d31-465f-abbd-c27862c512e2 │       v1.1.3 │      7.40 GB │ 2025-12-01T06:35:48.607464Z │
│ 1 │ c80a49c5-9823-4745-a9a0-265e3fb6320e │       v1.1.1 │      5.69 GB │ 2025-12-01T03:32:20.708202Z │
└───┴──────────────────────────────────────┴──────────────┴──────────────┴─────────────────────────────┘
```
### 1.6 Create version
```
model_space model version create --model-id <model-id> --version-name <version-name> 
```
Example result:
```
2025-12-02 16:17:10,985 [INFO] Creating new model version...
2025-12-02 16:17:11,834 [INFO] Create new version successfully
```
### 1.7 Delete version
```
model_space model version delete --model-id <model-id> --version-id <version-id> 
```
Example result:
```
Are you sure you want to delete version '5d7423df-aa9b-4342-be36-e47eec0693d9' of model '9ff0ee78-3c49-439a-a6df-bdd20acd63ff'? [y/N]: y
2025-12-02 16:36:11,060 [INFO] Version 5d7423df-aa9b-4342-be36-e47eec0693d9 is deleted!
```
### 1.8 Upload to version
```
model_space model upload --model-id <model-id> --version-id <version-id> --path <local-path> 
```
Example result:
```
2025-12-02 16:55:15,828 [INFO] Upload model from path huggingface_models/Qwen_Qwen2.5-VL-7B-Instruct
Uploading 16 files, total size: 15827.16 MB
Uploads: 0/16 completed                                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% • 0/16 bytes • ?          • -:--:--
Uploading: model-00001-of-00005.safetensors (3719.6MB) ━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━  43% • 1.7/3.9 GB • 54.4 MB/s  • 0:00:41
Uploading: model-00004-of-00005.safetensors (3685.7MB) ━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━  30% • 1.1/3.9 GB • 43.5 MB/s  • 0:01:03
Uploading: model-00003-of-00005.safetensors (3685.7MB) ━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  13% • 0.5/3.9 GB • 346.3 MB/s • 0:00:10
Uploading: model-00002-of-00005.safetensors (3685.7MB) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% • 0.0/3.9 GB • ?          • -:--:--

Finalizing uploaded files...
Uploads: 16/16 completed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 1/1 bytes • ? • 0:00:00

Upload summary: 16 files completed, 0 files failed

Successfully uploaded 16 files (15827.16 MB)
2025-12-02 17:06:29,122 [INFO] Successfully upload model into FPT Model Hub
```
### 1.9 Download model from version
```
model_space model download --model-id <model-id> --version-id <version-id> --path <local-path> 
```
Example result:
```
2025-12-02 17:15:43,609 [INFO] Download model to path download
2025-12-02 17:15:43,609 [INFO] Gathering Hub files info...
2025-12-02 17:15:43,609 [INFO] Get model versions detail from FPT Model Hub - Private Model
walking hub with prefix:  v2/
walking hub with prefix:  artifacts/
2025-12-02 17:15:45,157 [INFO] Done gather Hub files info.
Num retry: 0/3
Downloading 16 files, total size: 15827.16 MB
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/preprocessor_config.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model.safetensors.index.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/.gitattributes downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/tokenizer_config.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/README.md downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/generation_config.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/chat_template.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/config.json downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/merges.txt downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model-00005-of-00005.safetensors downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model-00002-of-00005.safetensors downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model-00004-of-00005.safetensors downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model-00003-of-00005.safetensors downloaded successfully!
File download/.space/9ff0ee78-3c49-439a-a6df-bdd20acd63ff/f626b820-7e0f-461b-9950-9d194eb7c2a1/model-00001-of-00005.safetensors downloaded successfully!
Download: 16/16 completed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 16/16 bytes • 6 bytes/s • 0:00:00

Download summary: 16 succeeded, 0 failed. Total attempted: 16 files.
Successfully downloaded 16 files (15827.16 MB)
2025-12-02 17:18:29,204 [INFO] Successfully download model from FPT Model Hub
```
---
## 2. Model Catalog
### 2.1 List model family
```
model_space model catalog family ls 
```
Example result:
```
2025-12-02 17:20:11,659 [INFO] Get model families from FPT Model Hub -Model Catalog
FPT Model Hub - 
 Model Catalog  
    Families    
┏━━━┳━━━━━━━━━━┓
┃   ┃ Name     ┃
┡━━━╇━━━━━━━━━━┩
│ 0 │ InternVL │
│ 1 │ Llama    │
│ 2 │ GPT      │
│ 3 │ Gemma    │
│ 4 │ Granite  │
│ 5 │ Deepseek │
│ 6 │ Qwen     │
│ 7 │ Mistral  │
└───┴──────────┘
```
### 2.2 List model catalog
```
model_space model catalog ls [--family <family>] 
```
Example result:
```
2025-12-02 17:22:37,806 [INFO] Get models list from FPT Model Hub - Model Catalog
                                  FPT Model Hub - Model Catalog                                  
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃   ┃ ID                                   ┃ Name                        ┃ Family ┃ Is Instruct ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━┩
│ 0 │ 37161f57-07f1-47c4-bcd9-e5072e156b66 │ google/gemma-3-1b-pt        │ Gemma  │ False       │
│ 1 │ cd626720-5a29-4d8c-8106-a3d5a0a8f224 │ google/gemma-3-4b-pt        │ Gemma  │ False       │
│ 2 │ 35a67c05-575d-4d9b-b10e-fe55695389f5 │ google/gemma-3-12b-pt       │ Gemma  │ False       │
│ 3 │ 7dedb11b-3c48-4f29-b900-c5e4082fd2f6 │ google/gemma-3-27b-pt       │ Gemma  │ False       │
│ 4 │ 2d1e2694-96b6-4c3e-b835-c9bbf6a85613 │ google/gemma-3-1b-it        │ Gemma  │ True        │
│ 5 │ 18f98c00-808e-47fe-a4be-3cb9992cade1 │ google/gemma-3-4b-it        │ Gemma  │ True        │
│ 6 │ 7079741c-8d67-4cd3-9081-2e8f8dec9365 │ google/gemma-3-12b-it       │ Gemma  │ True        │
│ 7 │ e82e67e4-e4a2-4413-8205-a6e46fa5fa5b │ google/gemma-3-27b-it       │ Gemma  │ True        │
│ 8 │ 1391ff72-0a31-4802-8f6c-57b46e29d66e │ google/medgemma-27b-text-it │ Gemma  │ True        │
└───┴──────────────────────────────────────┴─────────────────────────────┴────────┴─────────────┘
```
### 2.3 Download model catalog
```
model_space model catalog download --model-name <model-name> --path <local-path> 
```
Example result:
```
Download: 10/10 completed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 10/10 bytes • 0 bytes/s • 0:00:00

Download summary: 10 succeeded, 0 failed. Total attempted: 10 files.
2025-12-02 17:25:42,690 [INFO] Successfully download model from FPT Model Hub
```
---
### More References
For Python usage, see: [FPT Model Hub Python Reference](./python-sdk.md)  
For advanced features, see: [Advanced Usage](./advanced.md)  
For updates and release notes, see: [Changelog](./changelog.md)