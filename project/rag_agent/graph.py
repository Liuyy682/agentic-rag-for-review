from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .graph_state import State
from .nodes import *
from .edges import *

def create_agent_subgraph(llm, tools_list):
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    print("Compiling agent subgraph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("rerank_results", rerank_search_results)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)
    agent_builder.add_node("evaluate_answer", partial(evaluate_answer, llm=llm))

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
    agent_builder.add_edge("tools", "rerank_results")
    agent_builder.add_edge("rerank_results", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", "evaluate_answer")
    agent_builder.add_conditional_edges("evaluate_answer", route_after_answer_evaluation, {"orchestrator": "orchestrator", "__end__": END})

    return agent_builder.compile()

def create_agent_graph(llm, tools_list):
    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    agent_subgraph = create_agent_subgraph(llm, tools_list)

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    print("✓ Agent graph compiled successfully.")
    return agent_graph
