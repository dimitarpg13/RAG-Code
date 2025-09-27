from pydantic import BaseModel, Field
from app.core.clients import async_openai_client_obs
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
from app.indexing.github_parsing import CodeElement
import logging
import asyncio

logger = logging.getLogger(__name__)


# TODO: Implement the system prompt for deciding if a document is relevant or not.
SYSTEM_PROMPT = None

class DocFilterResult(BaseModel):
    """
    Result of the second-stage filtering step for a single document.
    """
    # TODO: Implement DocFilterResult for structured output. We need  two attributes:
    # - is_relevant: a boolean deciding if the document is relevant to the chat history
    # - extracted: a string that extracted the relevant text from the document if is_relevant is true.
    pass


class Selector:
    """Second-stage document selector that filters retrieved documents for relevance."""

    async def filter_doc(self, state: RetrieverAgentState, element: CodeElement) -> DocFilterResult:
        """Filter a single document for relevance to the chat history.
        
        Args:
            state: RetrieverAgentState containing chat_messages
            element: CodeElement to evaluate for relevance
            
        Returns:
            DocFilterResult with relevance decision and extracted content
            
        Raises:
            ConnectionError: If OpenAI API call fails
        """
        # TODO: Implement filter_doc that returns a DocFilterResult instance for each
        # document in new_documents. Use as input the system prompt, the chat history, and the document.
        raise NotImplemented
    
    async def filter_documents(self, state: RetrieverAgentState) -> list[DocFilterResult]:
        """Filter all new documents concurrently for relevance.
        
        Args:
            state: RetrieverAgentState containing new_documents to filter
            
        Returns:
            List of DocFilterResult objects for each document
        """
        # TODO: Implement filter_documents. Use asyncio.create_task and asyncio.gather to iterate 
        # in parallel through all the documents in new_documents with the filter_doc function.
        raise NotImplemented

    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState:
        """Main selector node - filters documents and updates retrieved_documents.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with relevant documents added to retrieved_documents
        """
        # TODO: In __call__:
        # - Create the filters with filter_documents
        # - append to state.retrieved_documents, the documents that are relevant 
        # after replacing the text of CodeElement by extracted in DocFilterResult.
        return state
    

selector = Selector()
