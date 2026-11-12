# RAG Multi-Agent with AWS Bedrock

PDF question answering using Titan embeddings, Pinecone, Claude (Bedrock), LangGraph and FastAPI.

## Run
1. pip install -r requirements.txt
2. Create .env with AWS and Pinecone keys
3. python ingest.py
4. python -m uvicorn main:app --reload