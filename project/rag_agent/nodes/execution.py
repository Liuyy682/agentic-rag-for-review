from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..graph_state import AgentState
from ..prompts import get_fallback_response_prompt, get_knowledge_fallback_prompt, get_task_executor_prompt


def task_executor(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    task_context = state.get("task_context", "").strip()
    sys_msg = SystemMessage(content=get_task_executor_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    task_context_injection = (
        [HumanMessage(content=f"[TASK CONTEXT]\n\n{task_context}")]
        if task_context else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL `rag_research` before answering this task.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + task_context_injection + [human_msg, force_search])
        tool_calls = response.tool_calls or []
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(tool_calls),
            "search_call_count": sum(1 for call in tool_calls if call.get("name") == "rag_research"),
            "iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + task_context_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "search_call_count": sum(1 for call in tool_calls if call.get("name") == "rag_research"),
        "iteration_count": 1,
    }


def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}


def knowledge_fallback_answer(state: AgentState, llm):
    prompt_content = (
        f"USER QUESTION:\n{state.get('question', '')}\n\n"
        "Answer using general knowledge. Do not include a Sources section."
    )
    response = llm.invoke([
        SystemMessage(content=get_knowledge_fallback_prompt()),
        HumanMessage(content=prompt_content),
    ])
    return {
        "messages": [response],
        "answer_mode": "knowledge_fallback",
        "used_knowledge_base": False,
        "fallback_triggered": True,
        "fallback_reason": state.get("fallback_reason") or state.get("retrieval_evidence_status") or "knowledge_base_unavailable",
        "retry_count_before_fallback": state.get("answer_evaluation_count", 0),
    }


def _sources_from_answer(answer: str) -> list[str]:
    if "**Sources:**" not in answer:
        return []
    source_block = answer.split("**Sources:**", 1)[1]
    sources = []
    for line in source_block.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if "." in cleaned:
            sources.append(cleaned)
    return sorted(set(sources))


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    answer_mode = state.get("answer_mode") or "rag_qa"
    used_knowledge_base = bool(state.get("used_knowledge_base", answer_mode != "knowledge_fallback"))
    sources = [] if answer_mode == "knowledge_fallback" else _sources_from_answer(str(answer))
    task_result = {
        "index": state["question_index"],
        "task_id": state.get("task_id") or f"task_{state['question_index'] + 1}",
        "question": state["question"],
        "answer": answer,
        "answer_mode": answer_mode,
        "used_knowledge_base": used_knowledge_base,
        "fallback_reason": state.get("fallback_reason", ""),
        "sources": sources,
        "diagnostics": {
            "answer_is_satisfactory": state.get("answer_is_satisfactory", False),
            "answer_evaluation_count": state.get("answer_evaluation_count", 0),
            "search_call_count": state.get("search_call_count", 0),
            "parent_retrieve_call_count": state.get("parent_retrieve_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "iteration_count": state.get("iteration_count", 0),
            "answer_mode": answer_mode,
            "fallback_triggered": state.get("fallback_triggered", False),
            "fallback_reason": state.get("fallback_reason", ""),
            "retrieval_evidence_status": state.get("retrieval_evidence_status", ""),
            "best_rerank_score": state.get("best_rerank_score"),
            "retry_count_before_fallback": state.get("retry_count_before_fallback", 0),
        },
    }
    return {
        "final_answer": answer,
        "answer_mode": answer_mode,
        "used_knowledge_base": used_knowledge_base,
        "task_results": [task_result],
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": answer,
            "answer_mode": answer_mode,
            "used_knowledge_base": used_knowledge_base,
            "fallback_reason": state.get("fallback_reason", ""),
            "sources": sources,
        }]
    }
