import boto3
from botocore.client import Config

s3 = boto3.client(
    's3',
    aws_access_key_id='your_aws_access_key_id',
    aws_secret_access_key='your_aws_secret_access_key',
    endpoint_url='your_endpoint_url',
    config=Config(signature_version='s3v4')
)

bucket_name = 'your_bucket_name'

file_paths = [
    "data/NLPLog_train.json",
    "data/NLPLog_val.json"
]

for file_path in file_paths:
    filename = file_path.split("/")[-1]
    s3_key = f'train-data/chatbot/{filename}'

    s3.upload_file(file_path, bucket_name, s3_key)
    print(f"File {file_path} uploaded to {bucket_name}/{s3_key}")