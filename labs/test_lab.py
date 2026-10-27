import json
from pathlib import Path
import tempfile
import unittest

from rag_lab import (Document, FIXTURES, LexicalRetriever, cosine, evaluate,
                     inspect_crud, load_documents, retrieval_metrics, split_document, terms)


class RetrievalLabTests(unittest.TestCase):
    def setUp(self):
        self.documents = load_documents(FIXTURES / "documents.json")
        self.chunks = [chunk for doc in self.documents for chunk in split_document(doc)]
        self.retriever = LexicalRetriever(self.chunks)

    def test_chunk_identity_and_original_offsets(self):
        doc = self.documents[0]
        first = split_document(doc, size=30, overlap=5)
        second = split_document(doc, size=30, overlap=5)
        self.assertEqual([c.id for c in first], [c.id for c in second])
        for chunk in first:
            self.assertEqual(chunk.text, doc.text[chunk.start:chunk.end])
        self.assertEqual(first[-1].end, len(doc.text))

    def test_chunk_parameters_reject_nonprogressing_window(self):
        with self.assertRaises(ValueError):
            split_document(self.documents[0], size=20, overlap=20)
        empty = Document("empty", "Empty", "", "v1", ("employee",))
        self.assertEqual(split_document(empty), [])

    def test_access_and_active_version_filter(self):
        results = self.retriever.search("培训补贴", {"employee"}, 100)
        self.assertTrue(results)
        self.assertTrue(all(r.chunk.active for r in results))
        self.assertNotIn("finance-budget", [r.chunk.document_id for r in results])
        self.assertTrue(all(r.chunk.version == "v2" for r in results
                            if r.chunk.document_id == "training"))
        self.assertEqual(self.retriever.search("培训补贴", set(), 100), [])

    def test_cosine_and_no_terms(self):
        self.assertAlmostEqual(cosine(terms("培训"), terms("培训")), 1.0)
        self.assertEqual(cosine(terms("!!!"), terms("培训")), 0.0)
        self.assertEqual(self.retriever.search("", {"employee"}), [])

    def test_hit_and_recall_count_distinct_documents(self):
        scores = retrieval_metrics({"a", "b", "c"}, ["a", "a"])
        self.assertEqual(scores["hit"], 1.0)
        self.assertAlmostEqual(scores["recall"], 1 / 3)
        with self.assertRaises(ValueError):
            retrieval_metrics(set(), [])

    def test_evaluation_uses_question_gold(self):
        rows = [{"id": "x", "question": "培训补贴", "gold_document_ids": ["training"]}]
        report = evaluate(self.retriever, rows, {"employee"}, 1)
        self.assertEqual(report["count"], 1)
        self.assertEqual(report["hit_at_k"], 1.0)
        self.assertEqual(report["recall_at_k"], 1.0)

    def test_inspect_reports_whole_file_and_error_lines(self):
        valid = {"ID": 1, "questions": "A question", "answers": "An answer", "news1": "Evidence"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            path.write_text(json.dumps(valid) + "\n\nnot json\n{}\n", encoding="utf-8")
            report = inspect_crud(path)
        self.assertEqual((report["valid"], report["blank"], report["invalid"]), (1, 1, 2))
        self.assertEqual([x["line"] for x in report["first_errors"]], [3, 4])


if __name__ == "__main__":
    unittest.main()
