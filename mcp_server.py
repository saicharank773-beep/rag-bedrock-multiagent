from mcp.server.fastmcp import FastMCP
from rag import retrieve

mcp = FastMCP("company-docs")

@mcp.tool()
def search_docs(query: str) -> str:
    """Search the company policy PDF. Use for refunds, leave, plans, support, privacy."""
    chunks = retrieve(query, k=3)
    return "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)

@mcp.tool()
def apply_student_discount(price: float) -> str:
    """Calculate price after the 20 percent student discount."""
    return f"Student price: {price * 0.8:.2f} rupees"

if __name__ == "__main__":
    mcp.run()  # stdio