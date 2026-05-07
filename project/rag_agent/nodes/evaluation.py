import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

import config
from ..graph_state import AgentState
from ..prompts import get_answer_evaluation_prompt
from ..schemas import AnswerEvaluation

DEFAULT_EVIDENCE_RERANK_SCORE_THRESHOLD = 0.0


def _collect_retrieved_context(state: AgentState) -> str:
    seen = set()
    context_parts = []

    context_summary = state.get("context_summary", "").strip()
    if context_summary:
        context_parts.append(f"## Compressed Research Context\n\n{context_summary}")

    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and msg.content and msg.content not in seen:
            seen.add(msg.content)
            tool_name = getattr(msg, "name", "tool")
            context_parts.append(f"## Tool Result: {tool_name}\n\n{msg.content}")

    return "\n\n".join(context_parts) if context_parts else "No retrieved context is available."


def _retrieval_evidence(state: AgentState) -> dict:
    tool_results = []
    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage) or getattr(msg, "name", "") != "rag_research":
            continue
        try:
            tool_results.append(json.loads(str(msg.content or "")))
        except json.JSONDecodeError:
            tool_results.append({
                "contexts": [],
                "gaps": ["RAG_RESEARCH_ERROR: unparseable tool result"],
                "diagnostics": {"evidence_status": "error"},
            })

    if not tool_results:
        return {
            "status": "insufficient",
            "reason": "no_retrieved_context",
            "best_rerank_score": None,
        }

    contexts = []
    gaps = []
    statuses = []
    scores = []
    errors = []
    for result in tool_results:
        contexts.extend(result.get("contexts") or [])
        gaps.extend(result.get("gaps") or [])
        diagnostics = result.get("diagnostics") or {}
        status = diagnostics.get("evidence_status")
        if status:
            statuses.append(str(status))
        if diagnostics.get("error"):
            errors.append(str(diagnostics.get("error")))
        score = diagnostics.get("best_rerank_score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
        for context in result.get("contexts") or []:
            score = context.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass

    best_score = max(scores) if scores else None
    threshold = config.RERANKER_SCORE_THRESHOLD
    if threshold is None and config.RERANKER_ENABLED:
        threshold = DEFAULT_EVIDENCE_RERANK_SCORE_THRESHOLD
    if contexts:
        if threshold is not None and best_score is not None and best_score < threshold:
            return {
                "status": "low_score",
                "reason": "best_rerank_score_below_threshold",
                "best_rerank_score": best_score,
            }
        return {
            "status": "sufficient",
            "reason": "",
            "best_rerank_score": best_score,
        }

    if "low_score" in statuses:
        return {
            "status": "low_score",
            "reason": "rerank_score_below_threshold",
            "best_rerank_score": best_score,
        }
    if errors or any(str(gap).startswith("RAG_RESEARCH_ERROR") for gap in gaps):
        return {
            "status": "error",
            "reason": "retrieval_error",
            "best_rerank_score": best_score,
        }
    return {
        "status": "insufficient",
        "reason": "no_relevant_document_context",
        "best_rerank_score": best_score,
    }


def _unsatisfactory_update(evidence: dict, critique: str, suggested_query: str):
    status = evidence.get("status") or "insufficient"
    reason = evidence.get("reason") or status
    feedback = (
        "[INTERNAL ANSWER EVALUATION]\n"
        "The draft answer is not ready to return from the knowledge base.\n\n"
        f"Critique:\n{critique}\n\n"
        f"Missing information:\n- {reason}\n\n"
        f"Suggested follow-up searches:\n- {suggested_query}\n\n"
        "Next step: call rag_research with a focused query for the missing information. "
        "If retry limits are exhausted, use knowledge_fallback."
    )
    return {
        "answer_is_satisfactory": False,
        "answer_evaluation_count": 1,
        "messages": [HumanMessage(content=feedback)],
        "retrieval_evidence_status": status,
        "best_rerank_score": evidence.get("best_rerank_score"),
        "fallback_reason": reason,
    }


def evaluate_answer(state: AgentState, llm):
    if state.get("answer_mode") == "knowledge_fallback":
        return {
            "answer_is_satisfactory": True,
            "answer_evaluation_count": 1,
            "used_knowledge_base": False,
        }

    answer = state.get("final_answer", "")
    evidence = _retrieval_evidence(state)
    if evidence["status"] != "sufficient":
        return _unsatisfactory_update(
            evidence,
            "The retrieved knowledge-base evidence is not sufficient to answer the question.",
            state.get("question", ""),
        )

    evaluation_input = (
        f"USER QUESTION:\n{state.get('question', '')}\n\n"
        f"RETRIEVED CONTEXT:\n{_collect_retrieved_context(state)}\n\n"
        f"DRAFT ANSWER:\n{answer}"
    )

    evaluation_response = llm.with_config(temperature=0).invoke([
        SystemMessage(content=get_answer_evaluation_prompt()),
        HumanMessage(content=evaluation_input),
    ])
    evaluation = _parse_answer_evaluation(str(evaluation_response.content))

    if evaluation.is_satisfactory:
        return {
            "answer_is_satisfactory": True,
            "answer_evaluation_count": 1,
            "retrieval_evidence_status": evidence["status"],
            "best_rerank_score": evidence.get("best_rerank_score"),
        }

    missing = "\n".join(f"- {item}" for item in evaluation.missing_information) or "- Unspecified gaps; reassess the question and retrieved context."
    suggested_queries = "\n".join(f"- {query}" for query in evaluation.suggested_search_queries) or "- Rephrase the original question to target the missing facts."
    feedback = (
        "[INTERNAL ANSWER EVALUATION]\n"
        "The draft answer is not ready to return.\n\n"
        f"Critique:\n{evaluation.critique}\n\n"
        f"Missing information:\n{missing}\n\n"
        f"Suggested follow-up searches:\n{suggested_queries}\n\n"
        "Next step: call rag_research with a focused query for the missing information, then produce a revised final answer. "
        "Do not return the same answer unless the new retrieval proves the missing information is unavailable."
    )
    return {
        "answer_is_satisfactory": False,
        "answer_evaluation_count": 1,
        "messages": [HumanMessage(content=feedback)],
        "retrieval_evidence_status": "insufficient",
        "best_rerank_score": evidence.get("best_rerank_score"),
        "fallback_reason": evaluation.critique or "llm_judged_evidence_insufficient",
    }


def _parse_answer_evaluation(content: str) -> AnswerEvaluation:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        raw = json.loads(match.group()) if match else {}

    return AnswerEvaluation(
        is_satisfactory=bool(raw.get("is_satisfactory", False)),
        critique=str(raw.get("critique", "")).strip(),
        missing_information=list(raw.get("missing_information") or []),
        suggested_search_queries=list(raw.get("suggested_search_queries") or []),
    )
