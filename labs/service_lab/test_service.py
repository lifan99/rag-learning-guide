"""Contract, HTTP, permission, and real adapter tests; no paid API calls."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from service import Config, DashScope, Engine, FIXTURES, ServiceError, bm25, create_app, evaluate, fake_vector, rrf, validate_answer


class LocalAPI(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.calls.append((self.path, body, self.headers.get("Authorization")))
        if self.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": [float(i + 1)] * body["dimensions"]} for i in range(len(body["input"]))]
            output = {"data": list(reversed(data)), "usage": {"total_tokens": 17}}
        elif self.path.endswith("/rerank"):
            output = {"output": {"results": [{"index": i, "relevance_score": .8} for i in reversed(range(len(body["input"]["documents"]))) ]}, "usage": {"total_tokens": 11}}
        elif self.path.endswith("/chat/completions"):
            answer = {"status": "answered", "answer": "年度上限 2000 元。", "citations": ["C1"]}
            output = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}], "usage": {"prompt_tokens": 23, "completion_tokens": 8}}
        else:
            self.send_error(500)
            return
        content = json.dumps(output).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.documents = Path(self.tmp.name) / "documents.json"
        self.documents.write_bytes((FIXTURES / "documents.json").read_bytes())
        self.cfg = Config(documents=self.documents, state=Path(self.tmp.name) / "state")
        self.engine = Engine(self.cfg)
        self.engine.ingest()

    def test_flask_complete_path_and_untrusted_identity(self):
        client = create_app(self.engine).test_client()
        self.assertEqual(client.get("/health").json["mode"], "fake")
        response = client.post("/ask", json={"question": "培训补贴需要什么条件？", "top_k": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "answered")
        self.assertEqual(response.json["retrieval"]["final_document_ids"], ["training"])
        self.assertIn("离线抽取演示", response.json["answer"])
        self.assertEqual(client.post("/ask", json={"question": "预算", "groups": ["finance"]}).status_code, 400)
        self.assertEqual(client.post("/ask", json={"question": "test", "top_k": True}).status_code, 400)
        self.assertEqual(client.post("/ask", json={"question": " "}).status_code, 400)

    def test_private_and_inactive_data_do_not_enter_index(self):
        manifest = json.loads(self.engine.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual({c["document_id"] for c in manifest["chunks"]}, {"training", "travel", "vpn"})
        self.assertEqual([c["version"] for c in manifest["chunks"] if c["document_id"] == "training"], ["v2"])

    def test_http_trace_ids_correlate_success_and_expected_failures(self):
        client = create_app(self.engine).test_client()
        private_query = "PRIVATE_QUERY_MUST_NOT_BE_LOGGED"
        with self.assertLogs("service", level="INFO") as logs:
            success = client.post("/ask", json={"question": private_query},
                                  headers={"X-Trace-ID": "untrusted-client-id"})
            bad_shape = client.post("/ask", json={"question": private_query, "groups": ["finance"]})
            bad_value = client.post("/ask", json={"question": " "})
            bad_json = client.post("/ask", data="{", content_type="application/json")
            self.engine.manifest_path.unlink()
            unavailable = client.post("/ask", json={"question": private_query})
        responses = [success, bad_shape, bad_value, bad_json, unavailable]
        self.assertEqual([r.status_code for r in responses], [200, 400, 400, 400, 503])
        trace_ids = [r.json["trace_id"] for r in responses]
        self.assertEqual(len(set(trace_ids)), len(responses))
        for response, trace_id in zip(responses, trace_ids):
            self.assertRegex(trace_id, r"^[0-9a-f]{32}$")
            self.assertEqual(response.headers["X-Trace-ID"], trace_id)
            matching = [record.getMessage() for record in logs.records if f"trace_id={trace_id} " in record.getMessage()]
            self.assertEqual(sum("rag_http " in message for message in matching), 1)
            if response in (success, bad_value, unavailable):
                self.assertEqual(sum("rag_ask " in message for message in matching), 1)
        log_text = "\n".join(logs.output)
        self.assertNotIn(private_query, log_text)
        self.assertNotIn("untrusted-client-id", log_text)
        self.assertNotIn(success.json["answer"], log_text)
        for source in success.json["sources"]:
            self.assertNotIn(source["text"], log_text)

    def test_request_entry_id_also_covers_health_and_http_errors(self):
        client = create_app(self.engine).test_client()
        for method, path, kwargs, status in (
            ("get", "/health", {}, 200),
            ("get", "/missing", {}, 404),
            ("post", "/ask", {"data": "x" * 16385, "content_type": "application/json"}, 413),
        ):
            with self.subTest(status=status):
                response = getattr(client, method)(path, **kwargs)
                self.assertEqual(response.status_code, status)
                self.assertRegex(response.json["trace_id"], r"^[0-9a-f]{32}$")
                self.assertEqual(response.headers["X-Trace-ID"], response.json["trace_id"])

    def test_business_entry_trace_id_survives_failure_without_payload_logging(self):
        private_error = "PRIVATE_PROVIDER_ERROR_MUST_NOT_BE_LOGGED"
        with self.assertLogs("service", level="INFO") as logs:
            with patch.object(self.engine, "_ask", side_effect=ServiceError(private_error)):
                with self.assertRaises(ServiceError) as raised:
                    self.engine.ask("private question")
        trace_id = raised.exception.trace_id
        self.assertRegex(trace_id, r"^[0-9a-f]{32}$")
        self.assertIn(f"trace_id={trace_id} ", logs.output[0])
        self.assertNotIn(private_error, "\n".join(logs.output))
        self.assertNotIn("private question", "\n".join(logs.output))

    def test_source_change_blocks_stale_index(self):
        docs = json.loads(self.documents.read_text(encoding="utf-8"))
        docs[0]["allowed_groups"] = ["finance"]
        self.documents.write_text(json.dumps(docs), encoding="utf-8")
        with self.assertRaisesRegex(ServiceError, "changed_run_ingest"):
            self.engine.ask("培训补贴")

    def test_invalid_citation_and_extra_fields(self):
        good = {"status": "answered", "answer": "回答", "citations": ["C1"]}
        self.assertEqual(validate_answer(good, {"C1": {}}), good)
        for bad in ({**good, "citations": ["C99"]}, {**good, "citations": []}, {**good, "extra": 1}):
            with self.assertRaises(ServiceError):
                validate_answer(bad, {"C1": {}})

    def test_permission_change_during_generation_blocks_old_result(self):
        documents = self.documents
        class UpdatingProvider:
            def generate(self, question, context):
                docs = json.loads(documents.read_text(encoding="utf-8"))
                for doc in docs:
                    if doc['id'] == 'training':
                        doc['allowed_groups'] = ['finance']
                documents.write_text(json.dumps(docs), encoding="utf-8")
                return {'status':'answered','answer':'old result',
                        'citations':[context[0]['citation']]}, {}
        live = Engine(replace(self.cfg,mode='live',retrieval='bm25'),UpdatingProvider())
        chunks, signature = live.snapshot()
        live.manifest_path.write_text(json.dumps({'signature':signature,'chunks':chunks}), encoding='utf-8')
        with self.assertRaisesRegex(ServiceError,'changed_during_request'):
            live.ask('培训补贴',1)

    def test_rrf_deduplicates_each_branch(self):
        self.assertEqual(rrf(["a", "a", "b"], ["b"]), ["b", "a"])

    def test_retrieval_modes_execute_only_required_branches(self):
        dense_engine = Engine(replace(self.cfg, retrieval="dense"))
        with patch("service.bm25", side_effect=AssertionError("dense must not execute BM25")):
            dense = dense_engine.ask("培训补贴", 1)
        self.assertEqual(dense["retrieval"]["bm25"], [])
        self.assertEqual(dense["retrieval"]["fused"], dense["retrieval"]["dense"])
        lexical_engine = Engine(replace(self.cfg, retrieval="bm25"))
        with patch("service.fake_vector", side_effect=AssertionError("BM25 must not execute dense embedding")):
            lexical = lexical_engine.ask("培训补贴", 1)
        self.assertEqual(lexical["retrieval"]["dense"], [])
        self.assertEqual(lexical["retrieval"]["fused"], lexical["retrieval"]["bm25"])
        simulated_rerank = Engine(replace(self.cfg, retrieval="dense", use_rerank=True))
        with patch("service.bm25", wraps=bm25) as ranker:
            simulated_rerank.ask("培训补贴", 1)
        ranker.assert_called_once()

    def test_eval_retains_rows_and_separates_modes(self):
        questions = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))
        with self.assertLogs("service", level="INFO") as logs:
            report = evaluate(self.engine, questions, 1)
        self.assertEqual(report["summary"]["mode"], "fake")
        self.assertEqual(len(report["rows"]), 4)
        self.assertEqual(report["rows"][-1]["recall"], .5)
        self.assertIn("usage", report["rows"][0]["result"])
        self.assertNotIn("answer_accuracy", report["summary"])
        for row in report["rows"]:
            self.assertEqual(row["trace_id"], row["result"]["trace_id"])
            self.assertTrue(any(f"trace_id={row['trace_id']} " in message for message in logs.output))

    def test_eval_records_input_and_service_failure_trace_ids(self):
        self.engine.manifest_path.unlink()
        questions = [
            {"id": "bad-input", "question": " ", "gold_document_ids": ["training"]},
            {"id": "unavailable", "question": "PRIVATE_EVAL_QUERY", "gold_document_ids": ["training"]},
        ]
        with self.assertLogs("service", level="INFO") as logs:
            report = evaluate(self.engine, questions, 1)
        self.assertEqual(report["summary"]["service_success_rate"], 0)
        self.assertEqual(len({r["trace_id"] for r in report["rows"]}), 2)
        for row in report["rows"]:
            self.assertIn("error", row)
            self.assertNotIn("result", row)
            self.assertRegex(row["trace_id"], r"^[0-9a-f]{32}$")
            self.assertTrue(any(f"trace_id={row['trace_id']} " in message for message in logs.output))
        self.assertNotIn("PRIVATE_EVAL_QUERY", "\n".join(logs.output))

    def test_live_failure_never_falls_back(self):
        class FailedProvider:
            def embed(self, texts):
                raise ServiceError("provider_unavailable")
        live = Engine(replace(self.cfg, mode="live"), FailedProvider())
        with self.assertRaisesRegex(ServiceError, "provider_unavailable"):
            live.ingest()
        self.assertFalse(live.manifest_path.exists())


class RESTAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LocalAPI.calls = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalAPI)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_actual_http_payloads_batch_order_and_usage(self):
        provider = DashScope(Config(base_url=self.url, rerank_url=self.url + "/rerank", api_key="local-test-only", dimensions=3))
        vectors, usages = provider.embed([f"text-{i}" for i in range(12)])
        self.assertEqual(len(vectors), 12)
        self.assertEqual(vectors[0], [1., 1., 1.])
        self.assertEqual(len(usages), 2)
        chunks = [{"title": "A", "text": "alpha"}, {"title": "B", "text": "beta"}]
        ranked, _ = provider.rerank("question", chunks)
        self.assertEqual(ranked, list(reversed(chunks)))
        result, usage = provider.generate("question", [{"citation": "C1", "text": "evidence"}])
        self.assertEqual(result["citations"], ["C1"])
        self.assertEqual(usage["completion_tokens"], 8)
        chat = LocalAPI.calls[-1]
        self.assertEqual(chat[1]["response_format"], {"type": "json_object"})
        self.assertFalse(chat[1]["enable_thinking"])
        self.assertEqual(chat[2], "Bearer local-test-only")
        self.assertEqual(LocalAPI.calls[-2][1]["parameters"]["top_n"], 2)

    def test_http_error_has_no_provider_body_or_secret(self):
        provider = DashScope(Config(base_url=self.url, api_key="local-test-only"))
        with self.assertRaisesRegex(ServiceError, "^provider_http_500$"):
            provider.post(self.url + "/fail", {})


@unittest.skipUnless(importlib.util.find_spec("pymilvus") and importlib.util.find_spec("milvus_lite"), "optional Milvus Lite dependency not installed")
class MilvusTests(unittest.TestCase):
    def test_real_database_with_deterministic_model_double(self):
        class ModelDouble:
            def embed(self, texts):
                return [fake_vector(text) for text in texts], []

            def generate(self, question, context):
                return {"status": "answered", "answer": "模型替身，用于数据库链路测试。", "citations": [context[0]["citation"]]}, {"stage": "generation", "model": "test-double"}

        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(Config(mode="live", dimensions=64, state=Path(folder)), ModelDouble())
            first = engine.ingest()
            second = engine.ingest()
            self.assertEqual(first["signature"], second["signature"])
            result = engine.ask("培训补贴需要哪些条件？", 1)
            self.assertEqual(result["retrieval"]["final_document_ids"], ["training"])
            self.assertEqual(result["sources"][0]["version"], "v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
