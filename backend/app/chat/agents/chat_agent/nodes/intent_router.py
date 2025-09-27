from pydantic import BaseModel, Field
from app.core.clients import async_openai_client_obs
from app.indexing.indexer import Indexer
from app.indexing.github_parsing import GitHubParser
from app.indexing.schemas import CodeElement
from app.chat.agents.chat_agent.state import ChatAgentState
import logging

logger = logging.getLogger(__name__)

# TODO: Implement your system prompt to guide the routing.
SYSTEM_PROMPT = None

class RouterDecision(BaseModel):
    # TODO: Add the Field utility for each and their related descriptions.
    needs_rag: bool
    query_vector_db: str | None


class IntentRouter:
    """
    Routes user queries to determine if RAG (Retrieval-Augmented Generation) is needed.
    
    This class analyzes the user's query and conversation history to decide whether
    the question can be answered directly or requires retrieving additional context
    from the knowledge base.
    """

    async def route(self, state: ChatAgentState) -> RouterDecision:
        """
        Determine routing decision for the user query.
        
        Args:
            state (ChatAgentState): Current conversation state with chat history
            
        Returns:
            RouterDecision: Contains needs_rag flag and optional search query
        """
        # TODO: Use the system prompt, the chat history, the structured output format RouterDecision,  
        # and get_samples, to implement the necessary messages to use as part of 
        # the async_openai_client_obs.responses.parse function.

        raise NotImplemented
    
    async def get_samples(self, state: ChatAgentState) -> list[CodeElement]:
        """
        Get sample documents from the knowledge base for routing context.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            List[str]: Sample document summaries to help with routing decision
        """
        # TODO: Implement get_samples to retrieve a few samples (e.g. 4) 
        # to demonstrate to the LLM what type of data is contained in the database.
        raise NotImplemented

    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the intent routing step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with routing decision
        """
        # TODO: Within __call__, call the route function and modify 
        # the need_rag and query_vector_db attribute of the state.
        return state
    

intent_router = IntentRouter()