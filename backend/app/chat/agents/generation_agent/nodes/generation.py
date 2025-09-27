from app.chat.agents.generation_agent.state import GenerationAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
import logging

logger = logging.getLogger(__name__)


# TODO: Implement your system prompt to generate an answer based on 
# the retrieved documents, the chat history, and the feedback, if it is not None.
# Make sure that the feedback is used to correct the behavior.
SYSTEM_PROMPT = None


class GeneratedAnswer(BaseModel):
    """Response model containing the generated answer and its source documents."""
    # TODO: Implement GeneratedAnswer for structured output. 
    # We need an answer attribute and a sources list attribute.
    pass

class Generator:
    """
    Generates AI responses using retrieved documents and conversation context.
    
    This class creates contextually appropriate responses by combining:
    1. Conversation history for context
    2. Retrieved document content for factual grounding
    3. Previous evaluation feedback for iterative improvement
    """

    async def generate(self, state: GenerationAgentState) -> GeneratedAnswer:
        """Generate an AI response using retrieved documents and conversation context.
        
        Args:
            state: GenerationAgentState with chat_messages, retrieved_documents, and optional feedback
            
        Returns:
            GeneratedAnswer containing the response and source file paths
            
        Raises:
            ConnectionError: If OpenAI API call fails
        """
        # TODO: Implement the generate function by using the system prompt, 
        # the chat history, the retrieved documents, and the feedback
        raise NotImplemented

    def check_sources(self, state: GenerationAgentState, answer: GeneratedAnswer) -> GeneratedAnswer:
        """Validate and filter source paths to only include actually retrieved documents.
        
        Args:
            state: GenerationAgentState containing retrieved_documents
            answer: GeneratedAnswer with potentially invalid source paths
            
        Returns:
            GeneratedAnswer with validated sources list
        """
        # TODO: Implement check_sources. It is very likely that the generate function misquoted the sources. Instead of providing wrong sources, 
        # let's filter the sources in GeneratedAnswer that do not belong exactly in [doc.source for doc in state.retrieved_documents]. 
        return answer

    async def __call__(self, state: GenerationAgentState) -> GenerationAgentState:
        """Main generation node - creates response and updates state.
        
        Args:
            state: Current GenerationAgentState
            
        Returns:
            Updated state with generation text, incremented iterations, and reset feedback
        """
        # TODO: In __call__,
        # - Use the generate function to generate an answer
        # - Use the check_sources function to ensure that the sources are correct
        # - Use the answer and sources attributes to update the generation attribute of the state
        # - Increment the num_iterations state attribute
        # - Reset the feedback attribute to None for the next iteration.
        return state
    

generator = Generator()