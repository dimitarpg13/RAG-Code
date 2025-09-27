from app.chat.agents.basic_rag.state import BasicChatAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
import logging

logger = logging.getLogger(__name__)

# TODO: Implement your system prompt for the generation
SYSTEM_PROMPT = None

# TODO: Implement GeneratedAnswer for structured output.
class GeneratedAnswer(BaseModel):
    """Response model for RAG-generated answers with source attribution.
    
    Attributes:
        answer: Generated response grounded in retrieved documents
        sources: List of file paths cited for facts used in the answer
    """
    answer = None
    sources = None


class Generator:
    """Answer generator component for RAG chat agent.
    
    Generates contextual responses by combining chat history with retrieved
    documents using OpenAI's language model and structured output parsing.
    """
    async def generate(self, state: BasicChatAgentState) -> GeneratedAnswer:
        """Generate a grounded answer using chat history and retrieved documents.
        
        Args:
            state: Current chat agent state with messages and retrieved documents
            
        Returns:
            GeneratedAnswer with response text and source citations
            
        Raises:
            ConnectionError: If OpenAI API call fails
            
        Note:
            Uses last 10 chat messages for context and formats retrieved documents
            as JSON for the language model to process and cite appropriately.
        """
        # TODO: Create a list of messages with:
        # - The system prompt
        # - The chat history: You can append the state.chat_messages to the list of messages,
        #  but it is good to add an indicator "###  Chat_history  ###:" before including that list.
        # - The retrieved documents:  Use the list state.retrieved_documents to generate 
        # one string that aggregates all the documents together. You can use the Pydantic 
        # function doc.model_dump_json(indent=2, exclude_none=True) to generate a string 
        # from one document. Again, it might be good to prepend this string of documents 
        # with "###  Documents  ###:\n{documents}", so the LLM is not confused.
        messages = []
        # TODO: Use the async_openai_client.responses.parse function to generate the response.
        raise NotImplemented
    
    async def __call__(self, state: BasicChatAgentState) -> BasicChatAgentState:
        """Execute generation step and update state with final response.
        
        Args:
            state: Current chat agent state
            
        Returns:
            Updated state with generation field containing formatted answer and sources
            
        Note:
            Formats the generated answer with sources list for display to user.
            This method makes the Generator callable as a node in the chat agent graph.
        """
        # TODO: In __call__, use the generate function to update the generation attribute of the state.
        return state


generator = Generator()