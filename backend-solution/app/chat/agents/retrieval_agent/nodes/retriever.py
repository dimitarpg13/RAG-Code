from app.indexing.indexer import Indexer
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
        indexer = Indexer(namespace=state.namespace)
        results = await indexer.search(state.queries[-1], max_results=20)
        return results
    
    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState:
        """Main retrieval node execution - searches and updates state.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with new_documents populated and num_iteration incremented
        """ 
        state.new_documents = await self.search(state)
        state.num_iteration += 1
        return state
    

retriever = Retriever()
