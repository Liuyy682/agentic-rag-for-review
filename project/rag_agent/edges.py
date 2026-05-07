from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ANSWER_EVALUATION_RETRIES, MAX_ITERATIONS, MAX_TOOL_CALLS

def route_after_intent(state: State) -> Literal["request_clarification", "chitchat_response", "rewrite_query"]:
    intent_type = state.get("intent_type", "")
    if intent_type == "chitchat":
        return "chitchat_response"
    if intent_type == "clarification" or not state.get("questionIsClear", False):
        return "request_clarification"
    return "rewrite_query"

def route_after_rewrite(state: State) -> Literal["request_clarification", "plan_rag_tasks"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    return "plan_rag_tasks"

def route_after_task_planning(state: State):
    return [
        Send(
            "task_executor",
            {
                "task_id": task.get("task_id") or f"task_{idx + 1}",
                "task_type": task.get("task_type", "rag_qa"),
                "question": task.get("query", ""),
                "question_index": idx,
                "original_query": task.get("original_query", state.get("originalQuery", "")),
                "task_context": task.get("context", ""),
                "messages": [],
            },
        )
        for idx, task in enumerate(state.get("task_plan", []))
        if task.get("query")
    ]

def route_after_task_executor_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    return route_after_task_executor_call(state)

def route_after_answer_evaluation(state: AgentState) -> Literal["task_executor", "knowledge_fallback", "__end__"]:
    if state.get("answer_mode") == "knowledge_fallback":
        return "__end__"

    if state.get("answer_is_satisfactory", False):
        return "__end__"

    if state.get("answer_evaluation_count", 0) >= MAX_ANSWER_EVALUATION_RETRIES:
        return "knowledge_fallback"

    if state.get("iteration_count", 0) >= MAX_ITERATIONS or state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "knowledge_fallback"

    return "task_executor"
