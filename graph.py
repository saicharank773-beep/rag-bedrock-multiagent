from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from rag import br, CLAUDE_ID, retrieve, answer

class State(TypedDict):
    question: str
    search_query: str
    chunks: List[dict]
    answer: str
    valid: bool
    attempts: int

def llm(prompt, max_tokens=300):
    r = br.converse(modelId=CLAUDE_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
    return r["output"]["message"]["content"][0]["text"].strip()

# Retrieval Agent: 2nd attempt la query rewrite pannum
def retrieval_node(s):
    q = s["search_query"]
    if s["attempts"] > 0:
        q = llm(f"Rewrite this as a better search query for a company policy document. "
                f"Return only the query.\n\nQuestion: {s['question']}", 60)
    return {"search_query": q, "chunks": retrieve(q, k=4)}

# Answer Agent
def answer_node(s):
    return {"answer": answer(s["question"], s["chunks"]), "attempts": s["attempts"] + 1}

# Validation Agent: answer context la irukka nu check
def validate_node(s):
    ctx = "\n".join(c["text"] for c in s["chunks"])
    verdict = llm(
        f"Context:\n{ctx}\n\nQuestion: {s['question']}\n\nAnswer:\n{s['answer']}\n\n"
        "Reply YES only if both are true: (1) every claim in the answer is supported by the context, "
        "and (2) if the answer says it doesn't know, the context really has no relevant information. "
        "If the answer says it doesn't know but the context DOES contain the answer, reply NO. "
        "Reply only YES or NO.", 5)
    return {"valid": "YES" in verdict.upper()}

def route(s):
    return "end" if s["valid"] or s["attempts"] >= 2 else "retry"

g = StateGraph(State)
g.add_node("retrieve", retrieval_node)
g.add_node("answer", answer_node)
g.add_node("validate", validate_node)
g.set_entry_point("retrieve")
g.add_edge("retrieve", "answer")
g.add_edge("answer", "validate")
g.add_conditional_edges("validate", route, {"end": END, "retry": "retrieve"})
workflow = g.compile()

def run(question):
    return workflow.invoke({"question": question, "search_query": question,
                            "chunks": [], "answer": "", "valid": False, "attempts": 0})

if __name__ == "__main__":
    for q in ["Refund edhana naal kulla kekkalaam?", "Company CEO yaaru?"]:
        r = run(q)
        print(f"\nQ: {q}\nA: {r['answer']}\nValid: {r['valid']} | Attempts: {r['attempts']}")