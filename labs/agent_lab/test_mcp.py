"""Run after installing requirements-mcp.txt; no cloud account is needed."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from agent_lab import Store
from mcp_client import inspect_server


class MCPIntegrationTests(unittest.TestCase):
    def test_timed_out_read_requests_cancellation_and_stops_agent(self):
        pending=Mock()
        pending.result.side_effect=TimeoutError()
        def schedule(coroutine,event_loop):
            coroutine.close()  # The injected future replaces this coroutine.
            return pending
        with tempfile.TemporaryDirectory() as tmp:
            with patch('mcp_client.asyncio.run_coroutine_threadsafe',side_effect=schedule):
                result=asyncio.run(inspect_server(agent=True,db=str(Path(tmp)/'timeout.sqlite3'),run_id='timeout'))
        pending.cancel.assert_called_once()
        self.assertEqual(result['results']['agent']['status'],'failed')
        self.assertEqual(result['results']['agent']['error_type'],'TimeoutError')
        self.assertIn('remaining time budget',result['results']['agent']['error'])

    def test_stdio_tools_feed_agent_and_local_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "mcp.sqlite3")
            result = asyncio.run(inspect_server(agent=True, db=db, run_id="integration"))
            definitions = {tool["name"]: tool["input_schema"] for tool in result["tools"]}
            self.assertEqual(set(definitions), {"lookup_policy", "lookup_employee"})
            self.assertTrue(all("employee_id" not in schema.get("properties", {})
                                for schema in definitions.values()))
            self.assertEqual(result["results"]["lookup_policy"]["source"], "training/v2")
            self.assertEqual(result["results"]["agent"]["status"], "awaiting_confirmation")
            store = Store(db)
            try:
                state = store.approve("integration", "employee001", True)
                self.assertEqual(state["status"], "completed")
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
