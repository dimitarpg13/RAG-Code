from dataclasses import dataclass
from langgraph.graph import END, StateGraph, START
from app.chat.agents.chat_agent.state import ChatAgentState

from app.chat.agents.chat_agent.nodes.intent_router import intent_router
from app.chat.agents.chat_agent.nodes.retrieval import retriever
from app.chat.agents.chat_agent.nodes.generation import generator
from app.chat.agents.chat_agent.nodes.simple_assistant import simple_assistant
from app.chat.agents.chat_agent.nodes.fallback import fallback


@dataclass(frozen=True)
class Nodes:
    """Node name constants for the chat agent graph."""
    INTENT_ROUTER = "intent_router"
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    SIMPLE_ASSISTANT = "simple_assistant"
    FALLBACK = "fallback"


def answer_type_router(state: ChatAgentState):
    """Route to RAG pipeline or simple assistant based on intent analysis."""
    if state.need_rag and state.query_vector_db:
        return Nodes.RETRIEVER
    else:
        return Nodes.SIMPLE_ASSISTANT
    

def empty_document_router(state: ChatAgentState):
    """Handle cases where retrieval returns no relevant documents."""
    if state.retrieved_documents:
        return Nodes.GENERATOR
    else:
        return Nodes.FALLBACK
    

def generation_evaluation_router(state: ChatAgentState):
    """Route based on response quality evaluation results."""
    if state.generation:
        return END
    else: 
        return Nodes.FALLBACK


# Build the agent graph with nodes and routing logic
builder = StateGraph(ChatAgentState)

# Add all processing nodes
builder.add_node(Nodes.INTENT_ROUTER, intent_router)
builder.add_node(Nodes.RETRIEVER, retriever.subagent)
builder.add_node(Nodes.GENERATOR, generator.subagent)
builder.add_node(Nodes.SIMPLE_ASSISTANT, simple_assistant)
builder.add_node(Nodes.FALLBACK, fallback)

# Define the conversation flow
builder.add_edge(START, Nodes.INTENT_ROUTER)

# Route based on whether RAG is needed
builder.add_conditional_edges(
    Nodes.INTENT_ROUTER, 
    answer_type_router,
    {
        Nodes.RETRIEVER: Nodes.RETRIEVER,
        Nodes.SIMPLE_ASSISTANT: Nodes.SIMPLE_ASSISTANT
    }
)

# Handle retrieval outcomes
builder.add_conditional_edges(
    Nodes.RETRIEVER, 
    empty_document_router,
    {
        Nodes.GENERATOR: Nodes.GENERATOR,
        Nodes.FALLBACK: Nodes.FALLBACK
    }
)

# Handle generation quality evaluation
builder.add_conditional_edges(
    Nodes.GENERATOR, 
    generation_evaluation_router,
    {
        END: END,
        Nodes.FALLBACK: Nodes.FALLBACK
    }
)

# Terminal nodes
builder.add_edge(Nodes.RETRIEVER, END) 
builder.add_edge(Nodes.SIMPLE_ASSISTANT, END) 
builder.add_edge(Nodes.GENERATOR, END) 
builder.add_edge(Nodes.FALLBACK, END) 

# Compile the agent for execution
chat_agent = builder.compile()