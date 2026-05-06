from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class TaskSpec(BaseModel):
    task_id: str = Field(
        description="Stable task identifier such as task_1."
    )
    task_type: Literal["rag_qa"] = Field(
        description="Execution type for the task. Only rag_qa is currently dispatched to task subgraphs."
    )
    query: str = Field(
        description="Self-contained task query to execute."
    )
    original_query: str = Field(
        description="Original user query before task normalization."
    )
    context: str = Field(
        default="",
        description="Minimal conversation context needed to execute this task."
    )
    constraints: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional execution constraints inferred from the user request."
    )

class IntentAnalysis(BaseModel):
    intent_type: Literal["rag_qa", "clarification", "chitchat", "follow_up"] = Field(
        description="User intent for the current message."
    )
    is_clear: bool = Field(
        description="Whether the current user message can be handled without asking for clarification."
    )
    original_query: str = Field(
        description="Original user message."
    )
    normalized_query: str = Field(
        description="Self-contained query after resolving follow-up context when possible."
    )
    clarification_needed: str = Field(
        default="",
        description="Clarification question to ask when intent or referents are unclear."
    )
    follow_up_context: str = Field(
        default="",
        description="Conversation context used to resolve a follow-up query."
    )
    tasks: List[TaskSpec] = Field(
        default_factory=list,
        description="RAG tasks to dispatch when the intent resolves to rag_qa."
    )

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )

class RetrievedContext(BaseModel):
    parent_id: str = Field(description="Parent chunk identifier.")
    source: str = Field(description="Source file name or path.")
    content: str = Field(description="Retrieved parent or child content.")
    score: Optional[float] = Field(default=None, description="Optional rerank or retrieval score.")

class RagResearchResult(BaseModel):
    query: str = Field(description="Executed retrieval query.")
    focus: str = Field(default="", description="Optional retrieval focus for this call.")
    contexts: List[RetrievedContext] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, object] = Field(default_factory=dict)

class TaskResult(BaseModel):
    index: int = Field(description="Original task order.")
    task_id: str = Field(description="Task identifier.")
    question: str = Field(description="Executed task query.")
    answer: str = Field(description="Final task answer.")
    diagnostics: Dict[str, object] = Field(default_factory=dict)

class AnswerEvaluation(BaseModel):
    is_satisfactory: bool = Field(
        description="Whether the answer fully and faithfully answers the question using the retrieved context."
    )
    critique: str = Field(
        description="Concise explanation of any answer quality problems, unsupported claims, or missing coverage."
    )
    missing_information: List[str] = Field(
        description="Specific missing facts or sub-questions that require more retrieval before answering."
    )
    suggested_search_queries: List[str] = Field(
        description="Targeted search queries to retrieve the missing information. Keep this empty if the answer is satisfactory."
    )
