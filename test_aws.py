import boto3, json
from dotenv import load_dotenv
load_dotenv()

REGION = "ap-southeast-2"
BUCKET = "rag-docs-saicharan-2026"
CLAUDE_ID = "au.anthropic.claude-haiku-4-5-20251001-v1:0"

# 1. S3
s3 = boto3.client("s3", region_name=REGION)
objs = s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])
print("S3 files:", [o["Key"] for o in objs])

# 2. Titan embedding
br = boto3.client("bedrock-runtime", region_name=REGION)
r = br.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "hello", "dimensions": 1024, "normalize": True}))
print("Embedding size:", len(json.loads(r["body"].read())["embedding"]))

# 3. Claude
resp = br.converse(
    modelId=CLAUDE_ID,
    messages=[{"role": "user", "content": [{"text": "Say hi in one line"}]}])
print("Claude:", resp["output"]["message"]["content"][0]["text"])