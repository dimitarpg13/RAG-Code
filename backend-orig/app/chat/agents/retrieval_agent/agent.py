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

# TODO: Instantiate the StateGraph with the RetrieverAgentState
builder = None
# TODO: Add the retriever, selector, and assessor nodes to the graph
# TODO: Add an edge from START to the retriever, from the retriever 
# to the selector, and from the selector to the assessor

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