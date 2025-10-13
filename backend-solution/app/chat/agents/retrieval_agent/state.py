from pydantic import BaseModel
from app.indexing.github_parsing import CodeElement


class RetrieverAgentState(BaseModel):
    chat_messages: list[dict[str, str]] = []
    namespace: str

    # The flag that the assessor can change if we need more data
    needs_rag: bool = False
    # The counter that captures the current number of retrieval iterations
    num_iteration: int = 0
    # The retrieved and selected documents up to the current iteration
    retrieved_documents: list[CodeElement] = []
    # The newly retrieved raw to documents that need to be assessed
    new_documents: list[CodeElement] = []
    # The list of queries that have been used up to now. Initialized to [query_vector_db]
    queries: list[str] = []