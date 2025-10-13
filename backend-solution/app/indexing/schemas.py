from pydantic import BaseModel, ConfigDict, field_serializer
from datetime import datetime


class IndexingRequest(BaseModel):
    github_url: str


class Repo(BaseModel):
    github_url: str
    namespace: str
    indexed_at: datetime
    # Allow Repo to be built directly from SQLAlchemy models instead of dicts
    model_config = ConfigDict(from_attributes=True)
    
    @field_serializer('indexed_at')
    def serialize_indexed_at(self, value: datetime) -> str:
        # Emit ISO 8601 strings so FastAPI's JSON responses stay ISO-formatted
        return value.isoformat()


class RepoListResponse(BaseModel):
    repos: list[Repo]


class File(BaseModel):
    content: str
    path: str
    extension: str


class CodeElement(BaseModel):
    text: str
    source: str
    header: str | None = None 
    extension: str
    description: str | None = None
