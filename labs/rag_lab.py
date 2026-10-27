"""A standard-library retrieval lab for the systematic RAG course."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    version: str
    allowed_groups: tuple[str, ...]
    active: bool = True


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    version: str
    title: str
    text: str
    start: int
    end: int
    allowed_groups: tuple[str, ...]
    active: bool


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def load_documents(path: Path) -> list[Document]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**{**row, "allowed_groups": tuple(row["allowed_groups"])})
            for row in rows]


def split_document(doc: Document, size: int = 100, overlap: int = 20) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    chunks = []
    start = 0
    while start < len(doc.text):
        end = min(start + size, len(doc.text))
        text = doc.text[start:end]
        identity = json.dumps(
            [doc.id, doc.version, size, overlap, start, end, text],
            ensure_ascii=False, separators=(",", ":"),
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        chunks.append(Chunk(chunk_id, doc.id, doc.version, doc.title, text,
                            start, end, doc.allowed_groups, doc.active))
        if end == len(doc.text):
            break
        start = end - overlap
    return chunks


def terms(text: str) -> Counter[str]:
    """Count Latin words, Han characters, and adjacent Han bigrams."""
    normalized = text.casefold()
    units = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        units.extend(run[i:i + 2] for i in range(len(run) - 1))
    return Counter(units)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.entries = [(chunk, terms(chunk.text)) for chunk in chunks]

    def search(self, query: str, groups: set[str], k: int = 3) -> list[SearchResult]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        query_vector = terms(query)
        results = []
        for chunk, vector in self.entries:
            if not chunk.active or not groups.intersection(chunk.allowed_groups):
                continue
            score = cosine(query_vector, vector)
            if score > 0:
                results.append(SearchResult(chunk, score))
        results.sort(key=lambda row: (-row.score, row.chunk.id))
        return results[:k]


def retrieval_metrics(gold: set[str], returned: list[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("retrieval metrics require nonempty gold evidence")
    matched = gold.intersection(returned)
    return {"hit": float(bool(matched)), "recall": len(matched) / len(gold)}


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, 1):
        chunk = result.chunk
        blocks.append(
            f"[C{number}] {chunk.title} | {chunk.document_id}/{chunk.version} "
            f"| chars {chunk.start}:{chunk.end}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def evaluate(retriever: LexicalRetriever, rows: list[dict], groups: set[str], k: int):
    details = []
    for row in rows:
        results = retriever.search(row["question"], groups, k)
        returned = [item.chunk.document_id for item in results]
        metrics = retrieval_metrics(set(row["gold_document_ids"]), returned)
        details.append({"id": row["id"], "returned": returned, **metrics})
    count = len(details)
    return {
        "count": count, "k": k,
        "hit_at_k": sum(row["hit"] for row in details) / count if count else None,
        "recall_at_k": sum(row["recall"] for row in details) / count if count else None,
        "questions": details,
    }


def inspect_crud(path: Path) -> dict:
    valid = blank = invalid = 0
    examples = []
    errors = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                blank += 1
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("expected an object")
                for name in ("questions", "answers", "news1"):
                    if not isinstance(item.get(name), str) or not item[name].strip():
                        raise ValueError(f"missing or invalid {name}")
            except (json.JSONDecodeError, ValueError) as exc:
                invalid += 1
                if len(errors) < 10:
                    errors.append({"line": line_number, "reason": str(exc)})
                continue
            valid += 1
            if len(examples) < 3:
                examples.append({
                    "question_id": f"{path.stem}:{line_number}:{item.get('ID', '')}",
                    "question": item["questions"],
                    "evidence_preview": item["news1"][:100],
                })
    return {"file": str(path), "valid": valid, "invalid": invalid, "blank": blank,
            "examples": examples, "first_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--top-k", type=int, default=3)
        cmd.add_argument("--size", type=int, default=100)
        cmd.add_argument("--overlap", type=int, default=20)
        cmd.add_argument("--group", action="append", default=None)
        if name == "demo":
            cmd.add_argument("--query", default="入职六个月可以申请培训补贴吗？")
    inspect = commands.add_parser("inspect-crud")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "inspect-crud":
        print(json.dumps(inspect_crud(args.path), ensure_ascii=False, indent=2))
        return
    documents = load_documents(FIXTURES / "documents.json")
    chunks = [chunk for doc in documents
              for chunk in split_document(doc, args.size, args.overlap)]
    retriever = LexicalRetriever(chunks)
    groups = set(args.group or ["employee"])
    if args.command == "evaluate":
        rows = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))
        print(json.dumps(evaluate(retriever, rows, groups, args.top_k),
                         ensure_ascii=False, indent=2))
    else:
        results = retriever.search(args.query, groups, args.top_k)
        print("Question:", args.query)
        print("Selected evidence:")
        print(build_context(results) or "No matching evidence")
        print("\nRanking:")
        for number, result in enumerate(results, 1):
            print(number, result.chunk.document_id, result.chunk.version,
                  round(result.score, 4), result.chunk.id)


if __name__ == "__main__":
    main()
