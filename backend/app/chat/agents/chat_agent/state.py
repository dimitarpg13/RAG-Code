from pydantic import BaseModel
from app.indexing.github_parsing import CodeElement


class ChatAgentState(BaseModel):
    # TODO: implement the agent state as we did for the basic RAG, but add 2 attributes:
    # - a `need_rag` boolean attribute to decide if we need RAG or not
    # - a `query_vector_db`` string attribute to capture the projection from chat_messages to a query we will use to retrieve data.
    pass