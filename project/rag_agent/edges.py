from typing import Literal
from langgraph.types import Send
from .graph_state import State, AgentState
from config import MAX_ANSWER_EVALUATION_RETRIES, MAX_ITERATIONS, MAX_TOOL_CALLS

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]
    
def route_after_orchestrator_call(state: AgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def route_after_answer_evaluation(state: AgentState) -> Literal["orchestrator", "__end__"]:
    if state.get("answer_is_satisfactory", False):
        return "__end__"

    if state.get("answer_evaluation_count", 0) >= MAX_ANSWER_EVALUATION_RETRIES:
        return "__end__"

    if state.get("iteration_count", 0) >= MAX_ITERATIONS or state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "__end__"

    return "orchestrator"
