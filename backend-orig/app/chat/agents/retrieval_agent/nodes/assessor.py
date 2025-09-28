from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
import logging

logger = logging.getLogger(__name__)


# TODO: Implement your system prompt to assess the current set of retrieved documents. 
# You will need to pass the chat history, the retrieved documents, and the previous queries.
SYSTEM_PROMPT = None

class ContextAssessment(BaseModel):
    """
    Decision emitted by the routing LLM.
    """
    # TODO:Implement ContextAssessment. We need two attributes:
    # - needs_rag: If true, we need more data
    # - query_vector_db: If needs_rag is true, then we need a different query than the previous ones to retrieve the necessary data. 


class ContextAssessor:
    """Assesses whether retrieved documents provide sufficient context to answer user queries."""

    async def assess(self, state: RetrieverAgentState) -> ContextAssessment:
        """Evaluate if current retrieved documents are sufficient for answering the query.
        
        Args:
            state: RetrieverAgentState with chat_messages, retrieved_documents, and queries
            
        Returns:
            ContextAssessment indicating if more retrieval is needed and potential new query
            
        Raises:
            Error: If OpenAI API call fails
        """
        # TODO: In the assess function, use the system prompt, the chat history, 
        # the retrieved documents, and the previous queries to assess the current set of data. 
        raise NotImplemented

    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState:
        """Main assessor node - evaluates context and updates state if more retrieval needed.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with needs_rag flag and new query if additional retrieval required
        """
        # TODO:  In __call__, run the assess function and if needs_rag is true and query_vector_db 
        # is not None, update the state's needs_rag and append query_vector_db to the state's queries.
        return state
    

assessor = ContextAssessor()