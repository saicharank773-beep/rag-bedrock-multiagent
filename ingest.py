import os, io, json, boto3
from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

REGION = "ap-southeast-2"
BUCKET = "rag-docs-saicharan-2026"
PDF_KEY = "company_policy.pdf"
INDEX_NAME = "docs"

s3 = boto3.client("s3", region_name=REGION)
br = boto3.client("bedrock-runtime", region_name=REGION)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Step 2: S3 la irundhu PDF eduthu text extract
def load_pdf():
    data = s3.get_object(Bucket=BUCKET, Key=PDF_KEY)["Body"].read()
    reader = PdfReader(io.BytesIO(data))
    return [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]

# Step 3: chunking (800 chars, 100 overlap)
def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# Step 4: Titan embedding
def embed(text):
    r = br.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}))
    return json.loads(r["body"].read())["embedding"]

# Step 5: Pinecone index create (illana already irukkum)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)

pages = load_pdf()
print("Pages:", len(pages))

vectors = []
for p in pages:
    for j, c in enumerate(chunk_text(p["text"])):
        vectors.append({
            "id": f"{PDF_KEY}-p{p['page']}-c{j}",
            "values": embed(c),
            "metadata": {"text": c, "page": p["page"], "source": PDF_KEY}})

print("Chunks:", len(vectors))
index.upsert(vectors=vectors)
print("Done! Stored in Pinecone.")