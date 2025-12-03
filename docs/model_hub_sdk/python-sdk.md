# FPT Model Hub SDK Python Usage

## 1. Private Model
### 1.1 Initialize the client with your token
```python
FPT_SPACE_URL = "https://ai-api.fptcloud.com"
FPT_SPACE_TOKEN = "your-personal-token"
TENANT_ID = "your-tenant-id"
```

```python
from model_space import ModelSpaceClient

client = ModelSpaceClient(fpt_space_url=FPT_SPACE_URL,
                          fpt_space_token=FPT_SPACE_TOKEN,
                          tenant_id=TENANT_ID)
```
### 1.2 List available models

```python
models = client.get_models()
print(models)
```
### 1.3 Create model
```python
model_name = "your-model-name"
model = client.create_model(model_name=model_name)
```
### 1.4 Delete model
```python
model_id = "your-model-id"
res = client.delete_model(model_id=model_id)
```
### 1.5 List version
```python
model_id = "your-model-id"
versions = client.get_model_version(model_id=model_id)
```
### 1.6 Create version
```python
model_id = "your-model-id"
version_name = "your-version-name"
version = client.create_model_version(model_id=model_id,
                                      model_version=version_name)
```
### 1.7 Delete version
```python
model_id = "your-model-id"
version_id = "your-version-id"
res = client.delete_model_version(model_id=model_id,
                                  version_id=version_id)
```
### 1.8 Upload to version
```python
model_id = "your-model-id"
version_id = "your-version-id"
local_path = "your/path/to/local/folder"
res = client.upload_model(model_id=model_id,
                          version_id=version_id,
                          local_directory=local_path)
```
### 1.9 Download model from version
```python
model_id = "your-model-id"
version_id = "your-version-id"
local_path = "your/path/to/local/folder"
res = client.download_model(model_id=model_id,
                            version_id=version_id,
                            local_directory=local_path)
```
---
## 2. Model Catalog
### 2.1 Initialize the client with your token
```python
FPT_SPACE_URL = "https://ai-api.fptcloud.com"
FPT_SPACE_TOKEN = "your-personal-token"
TENANT_ID = "your-tenant-id"
```

```python
from model_space import ModelCatalogClient

client = ModelCatalogClient(fpt_space_url=FPT_SPACE_URL,
                            fpt_space_token=FPT_SPACE_TOKEN,
                            tenant_id=TENANT_ID)
```
### 2.2 List model family
```python
families = client.get_model_families()
print(families)
```
### 2.3 List model catalog
```python
family = "Qwem"
models = client.get_models(family=family)
print(models)
```
### 2.4 Download model catalog
```python
model_name = "model-catalog-name"
res = client.download_model(model_name=model_name)
```
---
### More References
For CLI usage, see: [FPT Model Hub CLI Reference](./cli-reference.md)  
For advanced features, see: [Advanced Usage](./advanced.md)  
For updates and release notes, see: [Changelog](./changelog.md)
