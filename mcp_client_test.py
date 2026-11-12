import asyncio, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            r = await session.call_tool("search_docs", {"query": "refund policy"})
            print("\nsearch_docs:\n", r.content[0].text[:300])

            r = await session.call_tool("apply_student_discount", {"price": 699})
            print("\ndiscount:", r.content[0].text)

asyncio.run(main())