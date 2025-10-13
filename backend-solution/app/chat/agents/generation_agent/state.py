from pydantic import BaseModel
from app.indexing.github_parsing import CodeElement


class GenerationAgentState(BaseModel):

    chat_messages: list[dict[str, str]] = []
    generation: str | None = None
    retrieved_documents: list[CodeElement] = []
    is_valid: bool = True
    is_grounded: bool = True
    feedback: str | None = None
    num_iterations: int = 0