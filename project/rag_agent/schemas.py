from typing import List
from pydantic import BaseModel, Field

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
