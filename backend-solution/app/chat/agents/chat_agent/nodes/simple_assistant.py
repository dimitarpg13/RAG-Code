

from app.chat.agents.chat_agent.state import ChatAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client_obs
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a helpful assistant. Provide a answer to the user.
"""


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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])

        try:
            response = await async_openai_client_obs.responses.create(
                model='gpt-4.1-nano',
                input=messages,
                temperature=0.1,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

        return response.output_text
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the simple response generation step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with generated response
        """
        answer = await self.generate(state)
        state.generation = answer
        return state
    

simple_assistant = SimpleAssistant()