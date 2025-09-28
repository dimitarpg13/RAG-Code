from pydantic import BaseModel
from app.indexing.github_parsing import CodeElement


class GenerationAgentState(BaseModel):

    chat_messages: list[dict[str, str]] = []
    generation: str | None = None
    retrieved_documents: list[CodeElement] = []

    # TODO: We need the following additional attributes:
    # - is_valid: a boolean to capture if the generation is addressing the user's question,
    # - is_grounded: a boolean to capture if the generation is grounded in the retrieved documents
    # - feedback: a string to capture feedback to improve the generation in the next iteration.
    # - num_iterations: a counter to capture the number of times we went through the generation.
