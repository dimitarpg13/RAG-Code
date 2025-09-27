

from app.chat.agents.chat_agent.state import ChatAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client_obs
import logging

logger = logging.getLogger(__name__)

# TODO: Implement the system prompt.
SYSTEM_PROMPT = None

class SimpleAssistant:
    """
    Handles simple queries that don't require knowledge base retrieval.
    
    This class generates responses using only the LLM's pre-trained knowledge
    and conversation context, bypassing the RAG pipeline for efficiency
    when external knowledge is not needed.
    """

    async def generate(self, state: ChatAgentState) -> str:
        """
        Generate a simple response using conversation context only.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            str: Generated response text
        """
        # TODO: Use the chat history in async_openai_client_obs.responses.create 
        # to generate an answer.
        raise NotImplemented

    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the simple response generation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with generated response
        """
        # TODO:  In __call__, use the generate function to modify 
        # the generation attribute of the state.
        return state
    

simple_assistant = SimpleAssistant()