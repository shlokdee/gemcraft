"""Test SolidCraft's MCP server directly, no AI involved.

Run:  python test_mcp.py
Needs: pip install mcp
"""
import asyncio, base64, json, os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CLI = "./solidcraft-cli"          # <-- full path to the binary if it's not on your PATH
OUT = os.path.abspath("out")    # export + screenshot go here
os.makedirs(OUT, exist_ok=True)


def text_of(result):
    return "\n".join(c.text for c in result.content if c.type == "text")


async def main():
    server = StdioServerParameters(command=CLI, args=["mcp", "--allow-path", OUT])
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # 1. What tools does the server offer?
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])

            # 2. Build a 40x30x20 block with a 10mm hole through it, in ONE batch call.
            steps = [
                {"command": "sketch.create", "params": {"plane": "XY"}},
                {"command": "sketch.rectangle.two_point", "params": {"p0": [0, 0], "p1": [40, 30]}},
                {"command": "sketch.finish", "params": {}},
                {"command": "solid.extrude", "params": {"distance": 20}},
                {"command": "sketch.create", "params": {"plane": "XY"}},
                {"command": "sketch.circle.center", "params": {"center": [20, 15], "radius": 5}},
                {"command": "solid.extrude", "params": {"distance": 20, "operation": "cut"}},
            ]
            res = await s.call_tool("batch", {"commands": steps, "verbose": False})
            print("\nBATCH isError =", res.is_error)
            print(text_of(res)[:1500])

            # 3. Measure it. Expected volume: 40*30*20 - pi*5^2*20 = 22429.2 mm^3
            res = await s.call_tool("measure", {})
            print("\nMEASURE:", text_of(res)[:1500])

            # 4. Screenshot -> PNG file
            res = await s.call_tool("screenshot", {"view": "iso", "width": 800, "height": 600})
            for c in res.content:
                if c.type == "image":
                    path = os.path.join(OUT, "part.png")
                    open(path, "wb").write(base64.b64decode(c.data))
                    print("\nSaved screenshot:", path)
                elif c.type == "text":
                    print("screenshot info:", c.text[:300])

            # 5. Export STEP
            res = await s.call_tool("export", {"path": os.path.join(OUT, "part.step")})
            print("\nEXPORT:", text_of(res)[:300], "| isError =", res.is_error)


asyncio.run(main())