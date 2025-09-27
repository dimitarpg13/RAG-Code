from app.indexing.indexer import Indexer
from app.indexing.github_parsing import GitHubParser
from app.indexing.schemas import CodeElement
from app.chat.agents.basic_rag.state import BasicChatAgentState


class Retriever:
    """Retriever component for basic RAG chat agent.
    
    Handles document retrieval from indexed GitHub repositories
    using the search query from the last user message.
    """
    async def search(self, state: BasicChatAgentState) -> list[CodeElement]:
        """Search for relevant code elements based on user's latest message.
        
        Args:
            state: Current chat agent state containing GitHub URL and chat messages
            
        Returns:
            List of CodeElement objects most relevant to the user's query
            
        Note:
            Uses the last user message as search query. Configured for basic retrieval
            without filters or reranking for faster response times.
        """
        # TODO: - Let's implement the search function:
        # - Instantiate the Indexer
        # - Use the last user message from state.chat_messages as a query to the 
        # vector database.
        # - Retrieve the documents. For simplicity, you can prevent reranking and filtering.
        raise NotImplemented
    
    async def __call__(self, state: BasicChatAgentState) -> BasicChatAgentState: 
        """Execute retrieval step and update state with retrieved documents.
        
        Args:
            state: Current chat agent state
            
        Returns:
            Updated state with retrieved_documents field populated
            
        Note:
            This method makes the Retriever callable as a node in the chat agent graph.
        """ 
        # TODO: use the search function to update the retrieved_documents parameter of the state.
        return state

retriever = Retriever()