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


def answer_type_router(state: ChatAgentState) -> str:
    """Route to RAG pipeline or simple assistant based on intent analysis."""
    # TODO: Implement answer_type_router. If need_rag is true and query_vector_db exists, 
    # then we need to go to the retriever; otherwise, we move to the simple assistant.
    raise NotImplemented

def empty_document_router(state: ChatAgentState) -> str:
    """Handle cases where retrieval returns no relevant documents."""
    # TODO: Implement empty_document_router. If retrieved_documents is empty, 
    # we move to the fallback node; otherwise, we move to the generator.
    raise NotImplemented

def generation_evaluation_router(state: ChatAgentState) -> str:
    """Route based on response quality evaluation results."""
    # TODO: Implement generation_evaluation_router. If generation is None, 
    # we move to the fallback node; otherwise, we end the pipeline.
    raise NotImplemented


# Build the agent graph with nodes and routing logic
# TODO: Instantiate StateGraph with ChatAgentState.
builder = None

# Add all processing nodes
# TODO: Add the intent router, the retriever, the generator,
#  and the fallback nodes to the graph.

# Define the conversation flow
# TODO: Add an edge from START to the intent router.

# Route based on whether RAG is needed
# TODO: Add a conditional edge from the intent router using answer_type_router.

# Handle retrieval outcomes
# TODO: Add a conditional edge from the retriever using empty_document_router. 

# Handle generation quality evaluation
# TODO: Add a conditional edge from the generator using generation_evaluation_router.

# Terminal nodes
# TODO: End the pipeline after the fallback node and the simple assistant.

# Compile the agent for execution
chat_agent = builder.compile()