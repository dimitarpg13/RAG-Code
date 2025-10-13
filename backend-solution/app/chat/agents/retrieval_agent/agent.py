from langgraph.graph import END, StateGraph, START
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
from app.chat.agents.retrieval_agent.nodes.retriever import retriever
from app.chat.agents.retrieval_agent.nodes.selector import selector
from app.chat.agents.retrieval_agent.nodes.assessor import assessor
from dataclasses import dataclass


@dataclass(frozen=True)
class Nodes:
    RETRIEVER = "retriever"
    SELECTOR = 'selector'
    ASSESSOR = 'assessor'

builder = StateGraph(RetrieverAgentState)


builder.add_node(Nodes.RETRIEVER, retriever)
builder.add_node(Nodes.SELECTOR, selector)
builder.add_node(Nodes.ASSESSOR, assessor)

builder.add_edge(START, Nodes.RETRIEVER)
builder.add_edge(Nodes.RETRIEVER, Nodes.SELECTOR)
builder.add_edge(Nodes.SELECTOR, Nodes.ASSESSOR)

def need_more_context(state: RetrieverAgentState) -> str:
    if state.needs_rag and state.num_iteration <= 3:
        return Nodes.RETRIEVER
    else:
        return END
    
builder.add_conditional_edges(
    Nodes.ASSESSOR,
    need_more_context,
    {
        Nodes.RETRIEVER: Nodes.RETRIEVER,
        END: END
    }

)

retrieval_agent = builder.compile()
