# Advanced Usage for FPT Model Hub SDK

## HTTP Proxy Usage
For client internal environment, FPT Model Hub SDK support use SDK and interact with platform via Proxy IP

### 1. Proxy usage with CLI
export http environment variables:
```bash
export HTTP_PROXY=http://username:password@your-proxy-ip:port
```
or 
```bash
export HTTPS_PROXY=https://username:password@your-proxy-ip:port
```
or both HTTP_PROXY & HTTPS_PROXY
### 2. Proxy usage with Python Client
```python
FPT_SPACE_URL = "https://ai-api.fptcloud.com"
FPT_SPACE_TOKEN = "your-personal-token"
TENANT_ID = "your-tenant-id"
HTTP_PROXY = "http://username:password@your-proxy-ip:port"
HTTPS_PROXY = "http://username:password@your-proxy-ip:port"
```
for private model:
```python
from model_space import ModelSpaceClient

client = ModelSpaceClient(fpt_space_url=FPT_SPACE_URL,
                          fpt_space_token=FPT_SPACE_TOKEN,
                          tenant_id=TENANT_ID,
                          http_proxy=HTTPS_PROXY,
                          https_proxy=HTTPS_PROXY)
```
for model catalog:
```python
from model_space import ModelCatalogClient

client = ModelCatalogClient(fpt_space_url=FPT_SPACE_URL,
                            fpt_space_token=FPT_SPACE_TOKEN,
                            tenant_id=TENANT_ID,
                            http_proxy=HTTPS_PROXY,
                            https_proxy=HTTPS_PROXY)
```
---
### More References
For updates and release notes, see: [Changelog](./changelog.md)