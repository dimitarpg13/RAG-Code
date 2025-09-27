from pydantic import BaseModel
from app.indexing.schemas import CodeElement


class BasicChatAgentState(BaseModel):
    chat_messages: list[dict[str, str]] = []
    namespace: str | None = None
    generation: str | None = None
    retrieved_documents: list[CodeElement] = []