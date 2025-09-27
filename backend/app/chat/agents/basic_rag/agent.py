from dataclasses import dataclass
from langgraph.graph import END, StateGraph, START
from app.chat.agents.basic_rag.state import BasicChatAgentState

from app.chat.agents.basic_rag.nodes.retrieval import retriever
from app.chat.agents.basic_rag.nodes.generation import generator

@dataclass(frozen=True)
class Nodes:
    """Node name constants for the chat agent graph."""
    RETRIEVER = "retriever"
    GENERATOR = "generator"


# Create a state graph that manages the RAG pipeline flow
# BasicChatAgentState tracks: namespace, chat_messages, retrieved_documents, generation
builder = StateGraph(BasicChatAgentState)

# Add the two main processing nodes to the graph
builder.add_node(Nodes.RETRIEVER, retriever)  # Searches for relevant documents
builder.add_node(Nodes.GENERATOR, generator)  # Generates response using retrieved docs

# Define the linear flow: START → RETRIEVER → GENERATOR → END
builder.add_edge(START, Nodes.RETRIEVER)        # Entry point: start with document retrieval
builder.add_edge(Nodes.RETRIEVER, Nodes.GENERATOR)  # Pass retrieved docs to generator
builder.add_edge(Nodes.GENERATOR, END)          # Exit point: return generated response

# Compile the graph into an executable agent
# This creates a runnable that processes state through the defined workflow
basic_chat_agent = builder.compile()