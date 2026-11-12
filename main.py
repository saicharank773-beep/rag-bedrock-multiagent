from fastapi import FastAPI
from pydantic import BaseModel
from graph import run

app = FastAPI(title="RAG Multi-Agent API")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "RAG API running"}

@app.post("/ask")
def ask(q: Query):
    r = run(q.question)
    return {
        "answer": r["answer"],
        "route": r["route"],
        "valid": r["valid"],
        "attempts": r["attempts"],
        "sources": sorted({c["page"] for c in r["chunks"]}),
    }