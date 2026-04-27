import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    reference_answer: str
    source_file: str
    gold_parent_ids: List[str] = field(default_factory=list)
    gold_child_ids: List[str] = field(default_factory=list)
    gold_evidence_text: List[str] = field(default_factory=list)
    question_type: str = "unknown"
    difficulty: str = "unknown"
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], line_no: int) -> "EvalQuestion":
        required = ["question_id", "question", "reference_answer", "source_file"]
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ValueError(f"line {line_no}: missing required fields: {', '.join(missing)}")

        source_file = raw["source_file"]
        gold_source_files = raw.get("gold_source_files") or []
        if isinstance(source_file, list):
            source_file = source_file[0] if source_file else ""
        if gold_source_files and source_file not in gold_source_files:
            gold_source_files.insert(0, source_file)

        return cls(
            question_id=str(raw["question_id"]),
            question=str(raw["question"]),
            reference_answer=str(raw["reference_answer"]),
            source_file=str(source_file),
            gold_parent_ids=list(raw.get("gold_parent_ids") or []),
            gold_child_ids=list(raw.get("gold_child_ids") or []),
            gold_evidence_text=list(raw.get("gold_evidence_text") or []),
            question_type=str(raw.get("question_type") or "unknown"),
            difficulty=str(raw.get("difficulty") or "unknown"),
            tags=list(raw.get("tags") or []),
        )

    @property
    def gold_source_files(self) -> List[str]:
        return [self.source_file] if self.source_file else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "source_file": self.source_file,
            "gold_parent_ids": self.gold_parent_ids,
            "gold_child_ids": self.gold_child_ids,
            "gold_evidence_text": self.gold_evidence_text,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }


def load_eval_questions(path: str | Path) -> List[EvalQuestion]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    questions: List[EvalQuestion] = []
    seen_ids = set()
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            item = EvalQuestion.from_dict(raw, line_no)
            if item.question_id in seen_ids:
                raise ValueError(f"line {line_no}: duplicate question_id: {item.question_id}")
            seen_ids.add(item.question_id)
            questions.append(item)
    return questions


def dataset_stats(questions: Iterable[EvalQuestion]) -> Dict[str, Any]:
    rows = list(questions)
    by_type: Dict[str, int] = {}
    with_parent = 0
    with_child = 0
    with_evidence = 0
    for item in rows:
        by_type[item.question_type] = by_type.get(item.question_type, 0) + 1
        with_parent += int(bool(item.gold_parent_ids))
        with_child += int(bool(item.gold_child_ids))
        with_evidence += int(bool(item.gold_evidence_text))

    total = len(rows)
    return {
        "total": total,
        "question_types": by_type,
        "with_parent_ids": with_parent,
        "with_child_ids": with_child,
        "with_evidence_text": with_evidence,
        "parent_coverage": _ratio(with_parent, total),
        "child_coverage": _ratio(with_child, total),
        "evidence_coverage": _ratio(with_evidence, total),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0

