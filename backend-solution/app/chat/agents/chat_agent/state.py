from pydantic import BaseModel
from app.indexing.github_parsing import CodeElement


class ChatAgentState(BaseModel):

    chat_messages: list[dict[str, str]] = []
    namespace: str | None = None
    generation: str | None = None
    need_rag: bool = False
    query_vector_db: str | None = None
    retrieved_documents: list[CodeElement] = []