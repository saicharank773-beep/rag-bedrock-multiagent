from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from rag import br, CLAUDE_ID, retrieve, answer
from tool_calling import chat as tool_chat

class State(TypedDict):
    question: str
    route: str
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

# ---- Router (supervisor) agent ----
def router_node(s):
    label = llm(
        "Classify the user question into exactly one label.\n"
        "DOCS = question about company policy, refunds, leave, plans, prices, support, privacy.\n"
        "TOOLS = needs a price calculation such as a student discount.\n"
        "CHAT = greeting or small talk.\n"
        "The question may be in Tanglish. Reply with only the label.\n\n"
        f"Question: {s['question']}", 5).upper()
    route = "tools" if "TOOL" in label else "chat" if "CHAT" in label else "docs"
    print(f"  [ROUTER] -> {route}")
    return {"route": route}

# ---- Docs path: Retrieval, Answer, Validation agents ----
def retrieval_node(s):
    q = s["search_query"]
    if s["attempts"] > 0:
        q = llm("Rewrite this as a better search query for a company policy document. "
                f"Return only the query.\n\nQuestion: {s['question']}", 60)
    return {"search_query": q, "chunks": retrieve(q, k=4)}

def answer_node(s):
    return {"answer": answer(s["question"], s["chunks"]), "attempts": s["attempts"] + 1}

def validate_node(s):
    ctx = "\n".join(c["text"] for c in s["chunks"])
    verdict = llm(
        f"Context:\n{ctx}\n\nQuestion: {s['question']}\n\nAnswer:\n{s['answer']}\n\n"
        "Reply YES only if both are true: (1) every claim in the answer is supported by the context, "
        "and (2) if the answer says it doesn't know, the context really has no relevant information. "
        "If the answer says it doesn't know but the context DOES contain the answer, reply NO. "
        "Reply only YES or NO.", 5)
    return {"valid": "YES" in verdict.upper()}

def retry_route(s):
    return "end" if s["valid"] or s["attempts"] >= 2 else "retry"

# ---- Tool agent and Chat agent ----
def tool_node(s):
    return {"answer": tool_chat(s["question"]), "valid": True, "attempts": 1}

def chat_node(s):
    return {"answer": llm("Reply briefly and friendly. You help with company policy questions. "
                          f"Message: {s['question']}", 150), "valid": True, "attempts": 1}

# ---- Graph ----
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("retrieve", retrieval_node)
g.add_node("answer", answer_node)
g.add_node("validate", validate_node)
g.add_node("tool_agent", tool_node)
g.add_node("chat_agent", chat_node)

g.set_entry_point("router")
g.add_conditional_edges("router", lambda s: s["route"],
    {"docs": "retrieve", "tools": "tool_agent", "chat": "chat_agent"})
g.add_edge("retrieve", "answer")
g.add_edge("answer", "validate")
g.add_conditional_edges("validate", retry_route, {"end": END, "retry": "retrieve"})
g.add_edge("tool_agent", END)
g.add_edge("chat_agent", END)
workflow = g.compile()

def run(question):
    return workflow.invoke({"question": question, "route": "", "search_query": question,
                            "chunks": [], "answer": "", "valid": False, "attempts": 0})

if __name__ == "__main__":
    for q in ["Hello, epdi irukka?",
              "Refund edhana naal kulla kekkalaam?",
              "Premium plan student ku evlo aagum?",
              "Company CEO yaaru?"]:
        print(f"\nQ: {q}")
        r = run(q)
        print(f"A: {r['answer']}\nRoute: {r['route']} | Valid: {r['valid']} | Attempts: {r['attempts']}")