from rag import br, CLAUDE_ID, retrieve

# 1. Claude ku tools define pannu (menu maadhiri)
tools = {"tools": [
    {"toolSpec": {
        "name": "search_docs",
        "description": "Search the company policy PDF. Use for any question about refunds, leave, plans, support, privacy.",
        "inputSchema": {"json": {"type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"]}}}},
    {"toolSpec": {
        "name": "apply_student_discount",
        "description": "Calculate price after the 20 percent student discount.",
        "inputSchema": {"json": {"type": "object",
            "properties": {"price": {"type": "number", "description": "original monthly price in rupees"}},
            "required": ["price"]}}}},
]}

# 2. Tools oda real python functions
def search_docs(query):
    chunks = retrieve(query, k=3)
    return "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)

def apply_student_discount(price):
    return f"Student price: {price * 0.8:.2f} rupees"

FUNCS = {"search_docs": search_docs, "apply_student_discount": apply_student_discount}

# 3. Loop: Claude tool kekkum -> naam run pannuvom -> result thirumba kudukkuvom
def chat(question):
    messages = [{"role": "user", "content": [{"text": question}]}]
    for _ in range(5):  # max 5 rounds, infinite loop aagaadhu
        resp = br.converse(
            modelId=CLAUDE_ID, messages=messages, toolConfig=tools,
            system=[{"text": "Answer only from tool results. If not found, say I don't know. "
                             "Questions may be in Tanglish. Answer in simple English."}])
        msg = resp["output"]["message"]
        messages.append(msg)

        if resp["stopReason"] != "tool_use":
            return "".join(b.get("text", "") for b in msg["content"])

        results = []
        for block in msg["content"]:
            if "toolUse" in block:
                t = block["toolUse"]
                print(f"  [TOOL CALL] {t['name']}({t['input']})")
                out = FUNCS[t["name"]](**t["input"])
                results.append({"toolResult": {"toolUseId": t["toolUseId"],
                                               "content": [{"text": out}]}})
        messages.append({"role": "user", "content": results})
    return "Too many tool calls"

if __name__ == "__main__":
    for q in ["Hello, epdi irukka?",
              "Premium plan price enna?",
              "Premium plan student ku evlo aagum?",
              "Company CEO yaaru?"]:
        print(f"\nQ: {q}")
        print("A:", chat(q))