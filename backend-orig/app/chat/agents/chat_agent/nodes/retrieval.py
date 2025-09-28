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
        # TODO: In subagent,
        # - Initialize an initial RetrieverAgentState state from the current ChatAgentState.
        # - Call the ainvoke function on retrieval_agent.
        # - Cast the response from the ainvoke function as a RetrieverAgentState object.
        # - Update the retrieved_documents attribute of the ChatAgentState by using the 
        # retrieved_documents attribute of the final RetrieverAgentState.
        return state    
    
retriever = Retriever()