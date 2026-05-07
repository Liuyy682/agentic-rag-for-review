import json
import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from ..graph_state import State
from ..prompts import get_chitchat_prompt, get_intent_recognition_prompt, get_rewrite_query_prompt
from ..schemas import IntentAnalysis, QueryAnalysis


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


def _parse_query_analysis(content: str, fallback_query: str) -> QueryAnalysis:
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        raw = json.loads(match.group()) if match else {}

    questions = raw.get("questions") or []
    if isinstance(questions, str):
        questions = [questions]
    if not isinstance(questions, list):
        questions = []

    cleaned_questions = [str(q).strip() for q in questions if str(q).strip()]
    return QueryAnalysis(
        is_clear=bool(raw.get("is_clear", bool(cleaned_questions))),
        questions=cleaned_questions or [fallback_query],
        clarification_needed=str(raw.get("clarification_needed") or "").strip(),
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
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True,
            "intent_type": intent_type,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "normalized_query": response.normalized_query,
            "task_plan": [],
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
    query = state.get("normalized_query") or state.get("originalQuery") or ""
    conversation_summary = state.get("conversation_summary", "")

    prompt_input = (
        f"Conversation Context:\n{conversation_summary}\n\n"
        f"Original Query:\n{state.get('originalQuery', query)}\n\n"
        f"Normalized Query:\n{query}\n"
    )
    response_message = llm.with_config(temperature=0.1).invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=prompt_input),
    ])
    analysis = _parse_query_analysis(str(response_message.content), query)

    if not analysis.is_clear:
        clarification = analysis.clarification_needed or "I need more information to search the documents."
        return {
            "questionIsClear": False,
            "clarification_needed": clarification,
            "messages": [AIMessage(content=clarification)],
            "task_plan": [],
        }

    questions = analysis.questions[:3]
    tasks = [
        {
            "task_id": f"task_{idx + 1}",
            "task_type": "rag_qa",
            "query": question,
            "original_query": state.get("originalQuery", question),
            "context": conversation_summary,
            "constraints": {},
        }
        for idx, question in enumerate(questions)
    ]
    return {
        "questionIsClear": True,
        "rewrittenQuestions": questions,
        "task_plan": tasks,
    }


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
