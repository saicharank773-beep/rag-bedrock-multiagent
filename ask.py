import os, json, boto3
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
REGION = "ap-southeast-2"
CLAUDE_ID = "au.anthropic.claude-haiku-4-5-20251001-v1:0"

br = boto3.client("bedrock-runtime", region_name=REGION)
index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index("docs")

def embed(text):
    r = br.invoke_model(modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}))
    return json.loads(r["body"].read())["embedding"]

def retrieve(q, k=4):
    res = index.query(vector=embed(q), top_k=k, include_metadata=True)
    return [m["metadata"] for m in res["matches"]]

def ask(q):
    chunks = retrieve(q)
    ctx = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)
    prompt = f'Answer ONLY using the context. If not found, say "I don\'t know".\n\nContext:\n{ctx}\n\nQuestion: {q}'
    r = br.converse(modelId=CLAUDE_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0.2})
    return r["output"]["message"]["content"][0]["text"], sorted({c["page"] for c in chunks})

for q in ["Refund edhana naal kulla kekkalaam?",
          "Premium plan price enna?",
          "Company CEO yaaru?"]:
    a, pages = ask(q)
    print(f"\nQ: {q}\nA: {a}\nPages: {pages}")