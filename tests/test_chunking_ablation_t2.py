import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.runners.chunking_ablation import build_t2_retrieval_rows


class TestT2RetrievalAblationData(unittest.TestCase):
    def test_builds_rows_docs_and_deterministic_distractors(self):
        corpus = [
            {"_id": "d1", "title": "标题", "text": "第一段内容"},
            {"_id": "d2", "title": "", "text": "第二段内容"},
            {"_id": "d3", "title": "", "text": "第三段内容"},
        ]
        queries = [
            {"_id": "q1", "text": "第一个问题"},
            {"_id": "q2", "text": "第二个问题"},
        ]
        qrels = [
            {"qid": "q1", "pid": "d2", "score": 1},
            {"qid": "q2", "pid": "d1", "score": 1},
            {"qid": "q2", "pid": "d3", "score": 0},
        ]

        rows, source_docs = build_t2_retrieval_rows(
            corpus,
            queries,
            qrels,
            limit=1,
            offset=1,
            distractor_docs=1,
        )

        self.assertEqual(
            rows,
            [
                {
                    "question_id": "t2_q2",
                    "question": "第二个问题",
                    "gold_sentence_keys": ["d1"],
                    "gold_source_doc_ids": ["t2_doc_d1"],
                }
            ],
        )
        self.assertEqual([doc.doc_id for doc in source_docs], ["t2_doc_d1", "t2_doc_d2"])
        self.assertEqual(source_docs[0].text, "# 标题\n\n第一段内容")
        self.assertEqual(source_docs[0].sentence_keys, ("d1",))
        self.assertEqual(source_docs[1].sentence_texts, ("第二段内容",))


if __name__ == "__main__":
    unittest.main()
