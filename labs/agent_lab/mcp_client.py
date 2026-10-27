"""Start the official SDK's stdio transport and call the two read-only tools."""
import asyncio
import argparse
import json
from pathlib import Path
import sys
from concurrent.futures import TimeoutError as FutureTimeoutError

from mcp import Client, StdioServerParameters
from agent_lab import ScriptedDriver, Store, run_loop


async def inspect_server(agent=False, db="agent-mcp.sqlite3", run_id="mcp-lesson"):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )
    async with Client(parameters, read_timeout_seconds=10) as client:
        listed = await client.list_tools()
        definitions = [{"name": tool.name, "input_schema": tool.input_schema}
                       for tool in listed.tools]
        results = {}
        for name in ("lookup_policy", "lookup_employee"):
            result = await client.call_tool(name, {})
            if result.is_error:
                raise RuntimeError(f"MCP tool failed: {name}")
            results[name] = result.structured_content
        if agent:
            event_loop = asyncio.get_running_loop()

            async def remote_read(name, args):
                response = await client.call_tool(name, args)
                if response.is_error:
                    raise RuntimeError(f"MCP tool failed: {name}")
                return response.structured_content

            def run_agent():
                def read_tool(name, args, timeout=None):
                    limit = min(5.0, timeout) if timeout is not None else 5.0
                    if limit <= 0:
                        raise TimeoutError('No remaining budget for MCP read')
                    future = asyncio.run_coroutine_threadsafe(remote_read(name, args), event_loop)
                    try:
                        return future.result(timeout=limit)
                    except FutureTimeoutError:
                        future.cancel()  # Cooperative: not proof the remote action stopped.
                        raise TimeoutError('MCP read exceeded its remaining time budget') from None

                store = Store(db, read_tool=read_tool)
                try:
                    try:
                        store.load(run_id, "employee001")
                    except KeyError:
                        store.create(run_id)
                    return run_loop(store, run_id, "employee001", ScriptedDriver())
                finally:
                    store.close()

            results["agent"] = await asyncio.to_thread(run_agent)
        return {"tools": definitions, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="store_true", help="Run the planner using MCP read tools")
    parser.add_argument("--db", default="agent-mcp.sqlite3")
    parser.add_argument("--run-id", default="mcp-lesson")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(inspect_server(args.agent, args.db, args.run_id)),
                     ensure_ascii=False, indent=2))
