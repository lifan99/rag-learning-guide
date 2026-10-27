"""Tests focus on business outcomes and recovery, not textual snapshots."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_lab import (LiveDriver, POLICY, TOOLS, ScriptedDriver, Store, evaluate,
                       pack_context, run_loop, validate_call)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = Store(self.path / "state.sqlite3")
        self.store.create("r1")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def run_agent(self, **kwargs):
        return run_loop(self.store, "r1", "employee001", ScriptedDriver(), **kwargs)

    def test_draft_needs_approval_and_repeated_resume_is_idempotent(self):
        state = self.run_agent()
        self.assertEqual(state["status"], "awaiting_confirmation")
        self.assertEqual(state["draft"]["remaining_yuan"], 1400)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 0)
        first = self.store.approve("r1", "employee001", True)
        second = self.store.approve("r1", "employee001", True)
        self.assertEqual(first["application_id"], second["application_id"])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 1)

    def test_resume_after_business_commit_does_not_duplicate_write(self):
        self.run_agent()
        with self.assertRaises(RuntimeError):
            self.store.approve("r1", "employee001", True, crash_after_commit=True)
        self.store.close()
        self.store = Store(self.path / "state.sqlite3")
        state = self.store.approve("r1", "employee001", True)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 1)

    def test_decline_after_committed_crash_reconciles_completed_state(self):
        self.run_agent()
        with self.assertRaises(RuntimeError):
            self.store.approve("r1", "employee001", True, crash_after_commit=True)
        state = self.store.approve("r1", "employee001", False)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.store.load("r1", "employee001")["status"], "completed")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 1)

    def test_resume_after_tool_commit_reuses_journal(self):
        with self.assertRaises(InterruptedError):
            self.run_agent(crash_after_call=True)
        self.store.close()
        self.store = Store(self.path / "state.sqlite3")
        state = self.run_agent()
        self.assertEqual(state["status"], "awaiting_confirmation")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0], 3)

    def test_ownership_and_tool_arguments_are_not_model_controlled(self):
        with self.assertRaises(PermissionError):
            self.store.load("r1", "employee002")
        for name, args in [("lookup_employee", {"user_id": "employee002"}),
                           ("prepare_application", {"amount_yuan": True}),
                           ("delete_all", {})]:
            with self.assertRaises(ValueError):
                validate_call(name, args)

    def test_budget_stops_before_more_work(self):
        state = self.run_agent(max_steps=1)
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0], 1)
        self.store.create("zero-time")
        state = run_loop(self.store, "zero-time", "employee001", ScriptedDriver(), max_seconds=0)
        self.assertEqual(state["steps"], 0)

    def test_compaction_preserves_complete_pairs_and_facts(self):
        state = self.run_agent()
        baseline, meta = pack_context(state, 100000)
        small, reduced = pack_context(state, meta["context_bytes"] - 100)
        self.assertGreater(reduced["removed_pairs"], 0)
        self.assertIn(POLICY["source"], small[0]["content"])
        self.assertEqual((len(small) - 2) % 2, 0)
        for index in range(2, len(small), 2):
            self.assertEqual(small[index]["tool_calls"][0]["id"], small[index+1]["tool_call_id"])
        with self.assertRaises(ValueError):
            pack_context(state, 20)

    def test_late_tool_result_cannot_offer_confirmation(self):
        clock = [0.0]
        original = self.store.execute
        def late_prepare(state, call, timeout=None):
            result = original(state,call,timeout=timeout)
            if call['function']['name']=='prepare_application':
                clock[0] += 2.0
            return result
        with patch('agent_lab.time.monotonic',side_effect=lambda:clock[0]), \
             patch.object(self.store,'execute',side_effect=late_prepare):
            state = self.run_agent(max_seconds=1)
        self.assertEqual(state['status'],'budget_exhausted')
        self.assertEqual(len(state['history']),3)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM applications').fetchone()[0],0)
        with self.assertRaisesRegex(ValueError,'not awaiting confirmation'):
            self.store.approve('r1','employee001',True)

    def test_remaining_budget_reaches_external_read_tool(self):
        limits=[]
        def read_tool(name,args,timeout=None):
            limits.append(timeout)
            return POLICY.copy() if name=='lookup_policy' else {'months':14}
        self.store.read_tool=read_tool
        state=self.run_agent(max_seconds=3)
        self.assertEqual(state['status'],'awaiting_confirmation')
        self.assertEqual(len(limits),2)
        self.assertTrue(all(0 < t <= 3 for t in limits))

    def test_model_schema_matches_amount_validation_upper_bound(self):
        amount=next(t for t in TOOLS if t['function']['name']=='prepare_application')['function']['parameters']['properties']['amount_yuan']
        self.assertEqual(amount['maximum'],100000)

    def test_trace_contains_tool_parameters_without_api_key(self):
        trace = self.path / "trace.jsonl"
        self.run_agent(trace=trace)
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        prepared = next(r for r in records if r.get("tool") == "prepare_application")
        self.assertEqual(prepared["arguments"], {"amount_yuan": 900})
        self.assertNotIn("Authorization", trace.read_text())

    def test_approval_rechecks_budget_after_another_submission(self):
        self.run_agent()
        self.store.create("r2", amount=1000)
        run_loop(self.store, "r2", "employee001", ScriptedDriver())
        self.store.approve("r2", "employee001", True)
        with self.assertRaises(ValueError):
            self.store.approve("r1", "employee001", True)

    def test_new_draft_uses_balance_after_previous_run(self):
        self.run_agent()
        self.store.approve("r1", "employee001", True)
        self.store.create("second-run", amount=700)
        state = run_loop(self.store, "second-run", "employee001", ScriptedDriver())
        self.assertEqual(state["draft"]["remaining_yuan"], 500)
        self.assertFalse(state["draft"]["can_prepare"])
        self.assertEqual(state["status"], "rejected")
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM applications").fetchone()[0], 1)

    def test_live_adapter_uses_tool_call_protocol_and_actual_usage(self):
        seen = {}
        def fake_open(request, timeout):
            seen["body"] = json.loads(request.data)
            seen["url"] = request.full_url
            message, _ = ScriptedDriver().decide([], {"facts": {}}, timeout)
            return io.BytesIO(json.dumps({"choices": [{"message": message}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                                                   "total_tokens": 25}}).encode())
        driver = LiveDriver("https://provider.invalid/compatible-mode/v1", "dummy-test-key",
                            "qwen-plus", opener=fake_open)
        message, usage = driver.decide([{"role": "user", "content": "Hello"}], {}, 3)
        self.assertTrue(seen["url"].endswith("/chat/completions"))
        self.assertEqual(seen["body"]["tools"][0]["function"]["name"], "lookup_policy")
        self.assertFalse(seen["body"]["parallel_tool_calls"])
        self.assertEqual(usage["total_tokens"], 25)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup_policy")

    def test_missing_or_partial_usage_stays_unknown_in_cost(self):
        def fake_open(request, timeout):
            message, _ = ScriptedDriver().decide([], {"facts": {}}, timeout)
            return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())
        driver = LiveDriver("https://provider.invalid/v1", "dummy", opener=fake_open)
        _, usage = driver.decide([], {}, 3)
        self.assertEqual(usage, {"prompt_tokens": None, "completion_tokens": None,
                                 "total_tokens": None})

        class PartialUsageDriver(ScriptedDriver):
            mode = "live"
            def decide(self, messages, state, timeout):
                message, _ = super().decide(messages, state, timeout)
                counts = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
                if "employee" not in state["facts"] and "policy" in state["facts"]:
                    counts["completion_tokens"] = None
                return message, counts

        records = evaluate(PartialUsageDriver(), input_price=1, output_price=2)
        self.assertTrue(all(record["model_cost"] is None for record in records))
        self.assertTrue(all(record["model_tokens"]["completion_tokens"] is None for record in records))
        self.assertTrue(all(not record["model_usage_complete"] for record in records))

    def test_evaluation_checks_six_business_outcomes(self):
        records = evaluate()
        self.assertEqual(len(records), 6)
        self.assertTrue(all(record["passed"] for record in records))
        self.assertTrue(all(record["business_state_valid"] for record in records))
        self.assertTrue(all(record["trace_valid"] for record in records))

    def test_evaluation_detects_corrupt_business_rows_despite_legal_tools(self):
        original = Store.approve
        for mutation in ("owner", "amount", "run_id", "stored_id", "returned_id", "extra_row"):
            with self.subTest(mutation=mutation):
                def corrupt(store, run_id, owner, approve, **kwargs):
                    state = original(store, run_id, owner, approve, **kwargs)
                    if run_id == "normal":
                        statements = {
                            "owner": "UPDATE applications SET owner='employee003'",
                            "amount": "UPDATE applications SET amount_yuan=1",
                            "run_id": "UPDATE applications SET run_id='another-run'",
                            "stored_id": "UPDATE applications SET application_id='another-app'",
                            "extra_row": "INSERT INTO applications VALUES ('extra','extra-run','employee001',1)",
                        }
                        if mutation == "returned_id":
                            state["application_id"] = "wrong-returned-id"
                        else:
                            with store.db:
                                store.db.execute(statements[mutation])
                    return state
                with patch.object(Store, "approve", new=corrupt):
                    records = evaluate()
                normal = next(record for record in records if record["scenario"] == "normal")
                self.assertFalse(normal["passed"])
                self.assertFalse(normal["business_state_valid"])
                self.assertTrue(normal["trace_valid"])
                self.assertTrue(normal["status_valid"])
                self.assertTrue(all(record["passed"] for record in records if record is not normal))

    def test_evaluation_requires_zero_rows_for_every_rejected_scenario(self):
        original = Store.approve
        def bad_decline(store, run_id, owner, approve, **kwargs):
            state = original(store, run_id, owner, approve, **kwargs)
            if run_id == "human_declines":
                with store.db:
                    store.db.execute("INSERT INTO applications VALUES (?,?,?,?)",
                                     ("forbidden", "unrelated-run", owner, 900))
            return state

        def bad_rejection(store, run_id, owner, driver, **kwargs):
            state = run_loop(store, run_id, owner, driver, **kwargs)
            if run_id in {"insufficient_tenure", "missing_preapproval", "over_budget"}:
                with store.db:
                    store.db.execute("INSERT INTO applications VALUES (?,?,?,?)",
                                     ("forbidden", "unrelated-run", owner, 900))
            return state

        with patch.object(Store, "approve", new=bad_decline), \
             patch("agent_lab.run_loop", side_effect=bad_rejection):
            records = evaluate()
        rejected = [record for record in records if record["expected"]["status"] == "rejected"]
        self.assertEqual(len(rejected), 4)
        for record in rejected:
            self.assertFalse(record["passed"])
            self.assertFalse(record["business_state_valid"])
            self.assertTrue(record["trace_valid"])
            self.assertEqual(record["applications"], 1)

    def test_evaluation_checks_earlier_call_arguments_separately_from_business_rows(self):
        original = Store.approve
        def corrupt_journal(store, run_id, owner, approve, **kwargs):
            state = original(store, run_id, owner, approve, **kwargs)
            if run_id == "normal":
                with store.db:
                    store.db.execute("UPDATE calls SET arguments=? WHERE name='lookup_employee'",
                                     ('{"user_id":"employee003"}',))
            return state
        with patch.object(Store, "approve", new=corrupt_journal):
            record = evaluate()[0]
        self.assertFalse(record["passed"])
        self.assertFalse(record["trace_valid"])
        self.assertTrue(record["business_state_valid"])

    def test_evaluation_trace_survives_cleanup_and_omits_raw_model_material(self):
        secret = "SECRET-RAW-PROVIDER-MATERIAL"
        class NoisyDriver(ScriptedDriver):
            mode, model, output_tokens = "live", "fake-qwen", 128
            api_key = secret
            def decide(self, messages, state, timeout):
                message, _ = super().decide(messages, state, timeout)
                message["content"] = secret + "x" * 20000
                return message, {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
        trace = self.path / "evaluation.jsonl"
        records = evaluate(NoisyDriver(), trace=trace)
        saved = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(saved), 6)
        self.assertEqual(len({record["experiment_id"] for record in saved}), 1)
        self.assertTrue(all(record["passed"] for record in saved))
        self.assertNotIn(secret, trace.read_text(encoding="utf-8"))
        self.assertLess(trace.stat().st_size, 30000)
        for record, persisted in zip(records, saved):
            self.assertEqual(persisted["event"], "evaluation_result")
            self.assertEqual(persisted["run_id"], record["scenario"])
            self.assertEqual(persisted["mode"], "live")
            self.assertEqual(persisted["model"], "fake-qwen")
            self.assertEqual(persisted["config"]["output_tokens"], 128)
            self.assertEqual(persisted["config"]["max_steps"], 6)
            self.assertEqual(persisted["final_rows"], record["final_rows"])
            self.assertFalse(persisted["final_rows_truncated"])
            self.assertFalse(persisted["tool_actions_truncated"])
            self.assertEqual(len(persisted["tool_actions"]), 3)
            self.assertEqual(persisted["tool_actions"][-1]["arguments"], {"amount_yuan": record["expected"]["amount_yuan"]})
            self.assertIn("can_prepare", persisted["tool_actions"][-1]["result"])
            if record["expected"]["created"]:
                self.assertEqual(persisted["final_rows"], [{
                    "application_id": persisted["returned_application_id"],
                    "run_id": persisted["run_id"], "owner": persisted["expected"]["owner"],
                    "amount_yuan": persisted["expected"]["amount_yuan"]}])
            else:
                self.assertEqual(persisted["final_rows"], [])
        self.assertEqual(saved[-1]["errors"], [{"phase": "approval_commit", "type": "RuntimeError", "expected": True}])

    def test_evaluation_records_safe_model_and_approval_errors(self):
        secret = "SECRET-ERROR-BODY"
        class FailingDriver(ScriptedDriver):
            mode = "live"
            def decide(self, messages, state, timeout):
                raise RuntimeError(secret + "x" * 20000)
        trace = self.path / "failed-evaluation.jsonl"
        records = evaluate(FailingDriver(), trace=trace)
        self.assertTrue(all(not record["passed"] for record in records))
        self.assertTrue(all(record["stop_reason"] == "agent_error" for record in records))
        self.assertNotIn(secret, trace.read_text(encoding="utf-8"))
        self.assertEqual(records[0]["errors"], [{"phase": "agent", "type": "RuntimeError", "expected": False}])
        self.assertIsNone(records[0]["model_cost"])
        with patch.object(Store, "approve", side_effect=RuntimeError(secret)):
            records = evaluate(trace=trace)
        recovery = next(record for record in records if record["scenario"] == "resume_after_commit")
        self.assertFalse(recovery["passed"])
        self.assertEqual(recovery["stop_reason"], "approval_error")
        self.assertEqual(recovery["errors"], [{"phase": "approval", "type": "RuntimeError", "expected": False}])
        self.assertNotIn(secret, trace.read_text(encoding="utf-8"))

    def test_evaluate_cli_writes_requested_trace_and_honors_budgets(self):
        trace = self.path / "cli-evidence.jsonl"
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("agent_lab.py")),
                                 "evaluate", "--trace", str(trace), "--max-steps", "1"],
                                check=True, capture_output=True, text=True, encoding="utf-8")
        saved = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(saved), 6)
        self.assertTrue(all(record["config"]["max_steps"] == 1 for record in saved))
        self.assertTrue(all(record["stop_reason"] == "budget_exhausted" for record in saved))
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["summary"]["passed"], 0)


if __name__ == "__main__":
    unittest.main()
