from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    username: str
    namespace: str


class ChatResponse(BaseModel):
    response: str
