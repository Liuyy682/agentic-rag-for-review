from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *

def create_task_executor_subgraph(llm, tools_list):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    print("Compiling task executor subgraph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("task_executor", partial(task_executor, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node("knowledge_fallback", partial(knowledge_fallback_answer, llm=llm))
    agent_builder.add_node(collect_answer)
    agent_builder.add_node("evaluate_answer", partial(evaluate_answer, llm=llm))

    agent_builder.add_edge(START, "task_executor")
    agent_builder.add_conditional_edges("task_executor", route_after_task_executor_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "task_executor")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("knowledge_fallback", "collect_answer")
    agent_builder.add_edge("collect_answer", "evaluate_answer")
    agent_builder.add_conditional_edges("evaluate_answer", route_after_answer_evaluation, {"task_executor": "task_executor", "knowledge_fallback": "knowledge_fallback", "__end__": END})

    return agent_builder.compile()

def create_agent_subgraph(llm, tools_list):
    return create_task_executor_subgraph(llm, tools_list)

def create_agent_graph(llm, tools_list):
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    agent_subgraph = create_task_executor_subgraph(llm, tools_list)

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("recognize_intent", partial(recognize_intent, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("chitchat_response", partial(chitchat_response, llm=llm))
    graph_builder.add_node(plan_rag_tasks)
    graph_builder.add_node("task_executor", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "recognize_intent")
    graph_builder.add_conditional_edges("recognize_intent", route_after_intent)
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", END)
    graph_builder.add_edge("chitchat_response", END)
    graph_builder.add_conditional_edges("plan_rag_tasks", route_after_task_planning)
    graph_builder.add_edge(["task_executor"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer)

    print("✓ Agent graph compiled successfully.")
    return agent_graph
