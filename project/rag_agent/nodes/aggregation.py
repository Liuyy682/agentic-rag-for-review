from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..graph_state import State
from ..prompts import get_aggregation_prompt


def aggregate_answers(state: State, llm):
    answers = state.get("task_results") or state.get("agent_answers") or []
    if not answers:
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(answers, key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        answer_mode = ans.get("answer_mode", "rag_qa")
        used_knowledge_base = ans.get("used_knowledge_base", answer_mode != "knowledge_fallback")
        sources = ans.get("sources") or []
        formatted_answers += (
            f"\nAnswer {i}:\n"
            f"Question: {ans.get('question', '')}\n"
            f"answer_mode: {answer_mode}\n"
            f"used_knowledge_base: {used_knowledge_base}\n"
            f"sources: {sources}\n"
            f"{ans['answer']}\n"
        )

    conversation_memory = state.get("conversation_memory", "").strip()
    memory_section = f"Conversation memory:\n{conversation_memory}\n\n" if conversation_memory else ""
    user_message = HumanMessage(content=f"""{memory_section}Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}
