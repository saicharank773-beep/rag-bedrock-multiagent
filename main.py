from fastapi import FastAPI
from pydantic import BaseModel
from rag import retrieve, answer

app = FastAPI(title="RAG API")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "RAG API running"}

@app.post("/ask")
def ask(q: Query):
    chunks = retrieve(q.question)
    return {
        "answer": answer(q.question, chunks),
        "sources": sorted({c["page"] for c in chunks}),
    }