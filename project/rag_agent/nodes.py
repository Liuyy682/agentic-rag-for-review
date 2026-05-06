from typing import Literal, Set
import json
import logging
import re
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import AnswerEvaluation, IntentAnalysis
from .prompts import *
from utils import estimate_context_tokens
from rag_agent.reranker import get_reranker
import config
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR

logger = logging.getLogger(__name__)

def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {
        "conversation_summary": summary_response.content,
        "task_results": [{"__reset__": True}],
        "agent_answers": [{"__reset__": True}],
    }

def _task_dicts_from_intent(response: IntentAnalysis, fallback_query: str) -> list[dict]:
    if response.tasks:
        return [task.model_dump() for task in response.tasks[:3]]
    query = response.normalized_query or fallback_query
    return [{
        "task_id": "task_1",
        "task_type": "rag_qa",
        "query": query,
        "original_query": response.original_query or fallback_query,
        "context": response.follow_up_context or "",
        "constraints": {},
    }]

def _parse_intent_analysis(content: str, fallback_query: str) -> IntentAnalysis:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        raw = json.loads(match.group()) if match else {}

    intent_type = raw.get("intent_type") or "clarification"
    if intent_type not in {"rag_qa", "clarification", "chitchat", "follow_up"}:
        intent_type = "clarification"

    tasks = raw.get("tasks") or []
    if not isinstance(tasks, list):
        tasks = []

    return IntentAnalysis(
        intent_type=intent_type,
        is_clear=bool(raw.get("is_clear", False)),
        original_query=str(raw.get("original_query") or fallback_query),
        normalized_query=str(raw.get("normalized_query") or fallback_query),
        clarification_needed=str(raw.get("clarification_needed") or "").strip(),
        follow_up_context=str(raw.get("follow_up_context") or "").strip(),
        tasks=tasks,
    )

def recognize_intent(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    response_message = llm.with_config(temperature=0.1).invoke([SystemMessage(content=get_intent_recognition_prompt()), HumanMessage(content=context_section)])
    response = _parse_intent_analysis(str(response_message.content), last_message.content)
    intent_type = response.intent_type
    is_rag_intent = intent_type in ("rag_qa", "follow_up") and response.is_clear

    if is_rag_intent:
        tasks = _task_dicts_from_intent(response, last_message.content)
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True,
            "intent_type": intent_type,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "normalized_query": response.normalized_query,
            "rewrittenQuestions": [task["query"] for task in tasks],
            "task_plan": tasks,
            "task_results": [{"__reset__": True}],
            "agent_answers": [{"__reset__": True}],
        }

    if intent_type == "chitchat" and response.is_clear:
        return {
            "questionIsClear": True,
            "intent_type": "chitchat",
            "originalQuery": last_message.content,
            "normalized_query": response.normalized_query or last_message.content,
            "task_plan": [],
            "task_results": [{"__reset__": True}],
            "agent_answers": [{"__reset__": True}],
        }

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {
        "questionIsClear": False,
        "intent_type": "clarification",
        "clarification_needed": clarification,
        "messages": [AIMessage(content=clarification)],
        "task_plan": [],
    }

def rewrite_query(state: State, llm):
    return recognize_intent(state, llm)

def request_clarification(state: State):
    return {}

def plan_rag_tasks(state: State):
    task_plan = state.get("task_plan") or []
    if task_plan:
        return {
            "rewrittenQuestions": [task.get("query", "") for task in task_plan if task.get("query")],
        }

    query = state.get("normalized_query") or state.get("originalQuery") or ""
    if not query:
        return {"task_plan": []}

    task = {
        "task_id": "task_1",
        "task_type": "rag_qa",
        "query": query,
        "original_query": state.get("originalQuery", query),
        "context": state.get("conversation_summary", ""),
        "constraints": {},
    }
    return {"task_plan": [task], "rewrittenQuestions": [query]}

def chitchat_response(state: State, llm):
    query = state.get("normalized_query") or state.get("originalQuery") or state["messages"][-1].content
    response = llm.invoke([SystemMessage(content=get_chitchat_prompt()), HumanMessage(content=query)])
    return {"messages": [AIMessage(content=response.content)]}

def _parse_child_chunk_output(text: str) -> list[dict]:
    if not text:
        return []
    if text.startswith("NO_RELEVANT_CHUNKS") or text.startswith("RETRIEVAL_ERROR"):
        return []

    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("Parent ID: ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    parsed: list[dict] = []
    for block in blocks:
        parent_id = ""
        file_name = ""
        content_idx = None
        for idx, line in enumerate(block):
            if line.startswith("Parent ID: "):
                parent_id = line[len("Parent ID: "):].strip()
            elif line.startswith("File Name: "):
                file_name = line[len("File Name: "):].strip()
            elif line.startswith("Content:"):
                content_idx = idx
                break

        if content_idx is None:
            continue

        content_line = block[content_idx]
        content_first = content_line[len("Content:"):].lstrip()
        content_tail = block[content_idx + 1:]
        content = "\n".join([content_first] + content_tail).strip()

        extra_lines: list[str] = []
        for line in block:
            if line.startswith("Parent ID: "):
                continue
            if line.startswith("File Name: "):
                continue
            if line.startswith("Content:"):
                continue
            if line.startswith("Rerank Score:") or line.startswith("Rerank Rank:"):
                continue
            if not line.strip():
                continue
            extra_lines.append(line)

        parsed.append({
            "parent_id": parent_id,
            "file_name": file_name,
            "content": content,
            "extra_lines": extra_lines,
        })

    return parsed

def _format_child_chunk_output(docs: list[Document]) -> str:
    blocks: list[str] = []
    for doc in docs:
        metadata = doc.metadata or {}
        lines = [
            f"Parent ID: {metadata.get('parent_id', '')}",
            f"File Name: {metadata.get('source', '')}",
        ]
        extra_lines = metadata.get("_extra_lines") or []
        lines.extend(extra_lines)

        if "rerank_score" in metadata:
            try:
                lines.append(f"Rerank Score: {float(metadata.get('rerank_score')):.6f}")
            except (TypeError, ValueError):
                lines.append(f"Rerank Score: {metadata.get('rerank_score')}")
        if config.RETRIEVAL_DEBUG and "rerank_rank" in metadata:
            lines.append(f"Rerank Rank: {metadata.get('rerank_rank')}")

        content = doc.page_content or ""
        content_lines = content.splitlines()
        if content_lines:
            lines.append(f"Content: {content_lines[0]}")
            lines.extend(content_lines[1:])
        else:
            lines.append("Content: ")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)

def rerank_search_results(state: AgentState):
    if not config.RERANKER_ENABLED:
        return {}

    tool_calls = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = msg.tool_calls or []
            break

    if not tool_calls:
        return {}

    queries: list[str] = []
    query_by_call_id: dict[str, str] = {}
    for call in tool_calls:
        if call.get("name") != "search_child_chunks":
            continue
        query = (call.get("args") or {}).get("query")
        if query:
            queries.append(query)
            if call.get("id"):
                query_by_call_id[call["id"]] = query

    if not queries:
        return {}

    updates: list = []
    for msg in state["messages"]:
        if not isinstance(msg, ToolMessage) or getattr(msg, "name", "") != "search_child_chunks":
            continue

        tool_call_id = getattr(msg, "tool_call_id", None)
        query = None
        if tool_call_id and tool_call_id in query_by_call_id:
            query = query_by_call_id[tool_call_id]
        elif len(queries) == 1:
            query = queries[0]
        else:
            query = state.get("question", "")

        if not query:
            continue

        parsed = _parse_child_chunk_output(msg.content)
        if not parsed:
            continue

        docs: list[Document] = []
        for item in parsed[: config.RERANKER_TOP_N]:
            docs.append(
                Document(
                    page_content=item["content"],
                    metadata={
                        "parent_id": item["parent_id"],
                        "source": item["file_name"],
                        "_extra_lines": item["extra_lines"],
                    },
                )
            )

        top_k = min(len(docs), config.RERANKER_FINAL_TOP_K)
        if top_k <= 0:
            continue

        try:
            reranked = get_reranker().rerank(
                query=query,
                documents=docs,
                top_k=top_k,
                score_threshold=config.RERANKER_SCORE_THRESHOLD,
            )
        except Exception:
            logger.exception("Rerank failed; using original retrieval order")
            reranked = docs[:top_k]

        new_content = _format_child_chunk_output(reranked)
        updates.append(ToolMessage(content=new_content, name=msg.name, tool_call_id=tool_call_id, id=msg.id))

    return {"messages": updates} if updates else {}

# --- Task Executor Nodes ---
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

def orchestrator(state: AgentState, llm_with_tools):
    return task_executor(state, llm_with_tools)

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

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    search_calls = 0
    parent_retrieve_calls = 0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    parent_retrieve_calls += 1
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    search_calls += 1
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(
        update={
            "retrieval_keys": updated_ids,
            "search_call_count": search_calls,
            "parent_retrieve_call_count": parent_retrieve_calls,
        },
        goto=goto,
    )

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    task_result = {
        "index": state["question_index"],
        "task_id": state.get("task_id") or f"task_{state['question_index'] + 1}",
        "question": state["question"],
        "answer": answer,
        "diagnostics": {
            "answer_is_satisfactory": state.get("answer_is_satisfactory", False),
            "answer_evaluation_count": state.get("answer_evaluation_count", 0),
            "search_call_count": state.get("search_call_count", 0),
            "parent_retrieve_call_count": state.get("parent_retrieve_call_count", 0),
            "tool_call_count": state.get("tool_call_count", 0),
            "iteration_count": state.get("iteration_count", 0),
        },
    }
    return {
        "final_answer": answer,
        "task_results": [task_result],
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }

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

def evaluate_answer(state: AgentState, llm):
    answer = state.get("final_answer", "")
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
        return {"answer_is_satisfactory": True, "answer_evaluation_count": 1}

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
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    answers = state.get("task_results") or state.get("agent_answers") or []
    if not answers:
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(answers, key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}
