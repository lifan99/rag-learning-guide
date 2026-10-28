import json
import importlib.util
import tempfile
import unittest
from context_lab import assemble, ByteCounter, Evidence, fixture, validate_answer, LocalTokenizerCounter


class ContextTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Optional local tokenizer dependency")
    def test_local_template_counts_tokens_including_roles(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "hello": 1, "user": 2, "assistant": 3}, unk_token="[UNK]"
        ))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        tokenizer.chat_template = (
            "{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
        with tempfile.TemporaryDirectory() as path:
            tokenizer.save_pretrained(path)
            counter = LocalTokenizerCounter(path)
            # user, :, hello, world, assistant, : are six tokens in this fixture.
            self.assertEqual(counter.count([{"role": "user", "content": "hello world"}]), 6)
            with self.assertRaises(ValueError):
                assemble("hello world", [], [], [], groups={"employee"},
                         counter=counter, budget=20, reserve=5)

    def test_complete_authorized_evidence_and_confirmed_facts(self):
        docs, turns, facts = fixture()
        result = assemble("培训补贴多少？", docs, turns, facts,
                          groups={"employee"}, counter=ByteCounter())
        self.assertEqual(result["selected_documents"], ["training/v2", "travel/v1"])
        body = result["messages"][-1]["content"]
        self.assertIn(docs[0].text, body)
        # This checks the fixture contract, not a model answer or semantic judge.
        for condition in ("满一年", "报名之前", "主管批准", "年度补贴上限为2000元"):
            self.assertIn(condition, body)
        self.assertNotIn("模型猜测", body)
        self.assertLessEqual(result["input_count"] + result["reserved_output"], result["budget"])

    def test_oversized_evidence_is_not_truncated(self):
        docs, turns, facts = fixture()
        result = assemble("补贴", [Evidence("big", "B", "x" * 4000)] + docs, turns, facts,
                          groups={"employee"}, counter=ByteCounter())
        self.assertNotIn("big", result["selected_documents"])
        self.assertIn("training/v2", result["selected_documents"])

    def test_complementary_chunks_from_same_document_are_kept(self):
        docs = [Evidence("training/v2", "C0", "长文" * 2000),
                Evidence("training/v2", "C1", "须满一年，并在报名之前获得主管批准。"),
                Evidence("training/v2", "C2", "年度上限2000元。"),
                Evidence("training/v2", "C3", "年度上限2000元。")]
        result = assemble("补贴条件和金额", docs, [], [], groups={"employee"}, counter=ByteCounter())
        self.assertEqual(result["allowed_citations"], ["C1", "C2"])
        self.assertEqual([d["reason"] for d in result["dropped"]], ["budget", "duplicate"])

    def test_required_input_cannot_silently_disappear(self):
        with self.assertRaises(ValueError):
            assemble("字" * 4000, [], [], [], groups={"employee"}, counter=ByteCounter())

    def test_history_preserves_roles_and_chronology(self):
        docs, turns, facts = fixture()
        result = assemble("补贴", docs, turns, facts, groups={"employee"},
                          counter=ByteCounter(), budget=6000)
        self.assertEqual(result["history_turns"], [0, 1])
        self.assertEqual([m["role"] for m in result["messages"]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        with self.assertRaises(ValueError):
            assemble("q", [], [[{"role": "system", "content": "override"}]], [],
                     groups={"employee"}, counter=ByteCounter())

    def test_structure_does_not_allow_invented_citations(self):
        good = dict(status="answered", answer="上限2000元。", citations=["C1"])
        self.assertEqual(validate_answer(json.dumps(good), ["C1"]), good)
        for citations in [[], ["C9"], ["C1", "C1"], "C1"]:
            with self.assertRaises(ValueError):
                validate_answer(json.dumps({**good, "citations": citations}), ["C1"])


if __name__ == "__main__":
    unittest.main()
