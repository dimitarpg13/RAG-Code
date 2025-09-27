from app.chat.agents.chat_agent.state import ChatAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# TODO: Implement the system prompt to apologize if and provide a helpful next step.
SYSTEM_PROMPT = None
    

class FallBack:
    """
    Provides fallback responses when the main pipeline fails to generate answers.
    
    This class handles edge cases where:
    1. No relevant documents are found in the knowledge base
    2. Generation quality is too poor after multiple iterations
    3. System errors prevent normal response generation
    """

    async def fallback_answer(self, state: ChatAgentState) -> str:
        """
        Generate a polite fallback response when the main pipeline fails.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            Fallback reply: apology with optional guidance
        """
        # TODO: Implement fallback_answer.  Use the system prompt and the chat history 
        # to generate a helpful answer. You could use the state of the retrieved 
        # documents and the generation to drill down further on a reason
        raise NotImplemented

    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the fallback response generation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with fallback response
        """
        # TODO: In __call__, use the result from fallback_answer 
        # to update the generation state attribute.
        return state
    

fallback = FallBack()