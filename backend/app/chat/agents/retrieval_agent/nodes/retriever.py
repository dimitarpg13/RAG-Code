from app.indexing.indexer import Indexer
from app.indexing.github_parsing import GitHubParser
from app.indexing.schemas import CodeElement
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves code elements from GitHub repositories using search queries."""
    
    async def search(self, state: RetrieverAgentState) -> list[CodeElement]:
        """Search for code elements using the most recent query from state.
        
        Args:
            state: RetrieverAgentState containing namespace and queries
            
        Returns:
            List of up to 20 CodeElement objects matching the search query
        """
        # TODO: retrieve 20-30 documents based on the last entry in state.queries
        raise NotImplemented
    
    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState: 
        """Main retrieval node execution - searches and updates state.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with new_documents populated and num_iteration incremented
        """ 
        # TODO: In __call__, update the new_documents with the retrieved 
        # documents and increment num_iteration by 1.
        return state
    

retriever = Retriever()