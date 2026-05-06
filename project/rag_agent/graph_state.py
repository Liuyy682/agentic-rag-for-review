from typing import Annotated, List, Set
from langgraph.graph import MessagesState
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

class State(MessagesState):
    """State for main agent graph"""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = "" 
    rewrittenQuestions: List[str] = []
    intent_type: str = ""
    normalized_query: str = ""
    clarification_needed: str = ""
    task_plan: List[dict] = []
    task_results: Annotated[List[dict], accumulate_or_reset] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """State for individual task executor subgraph"""
    task_id: str = ""
    task_type: str = "rag_qa"
    question: str = ""
    question_index: int = 0
    original_query: str = ""
    task_context: str = ""
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    research_results: List[dict] = []
    kept_parent_ids: List[str] = []
    excluded_parent_ids: List[str] = []
    final_answer: str = ""
    task_results: List[dict] = []
    agent_answers: List[dict] = []
    answer_is_satisfactory: bool = False
    answer_evaluation_count: Annotated[int, operator.add] = 0
    search_call_count: Annotated[int, operator.add] = 0
    parent_retrieve_call_count: Annotated[int, operator.add] = 0
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
