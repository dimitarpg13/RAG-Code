from app.chat.agents.chat_agent.state import ChatAgentState
from app.chat.agents.retrieval_agent.agent import retrieval_agent
from app.chat.agents.retrieval_agent.state import RetrieverAgentState


class Retriever:
    """Retrieval node that invokes the retrieval agent as a subagent."""

    async def subagent(self, state: ChatAgentState) -> ChatAgentState:
        """Execute retrieval subagent to find relevant documents.
        
        Args:
            state: ChatAgentState containing namespace, query_vector_db, and chat_messages
            
        Returns:
            Updated ChatAgentState with retrieved_documents populated from subagent
        """
        initial_retriever_state = RetrieverAgentState(
            namespace=state.namespace,
            queries=[state.query_vector_db],
            chat_messages=state.chat_messages
        )
        response = await retrieval_agent.ainvoke(
            initial_retriever_state.model_dump()
        )
        final_retriever_state = RetrieverAgentState(**response)
        state.retrieved_documents = final_retriever_state.retrieved_documents
        return state    
    
retriever = Retriever()