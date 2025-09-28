from fastapi import APIRouter, Depends, HTTPException
from app.core.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.indexing.crud import get_indexed_repos
from app.indexing.tasks import run_indexing_task

from app.indexing.schemas import (
    IndexingRequest, 
    RepoListResponse,
    Repo
)
import logging


logger = logging.getLogger(__name__)

# First, we instantiate a sub-router to collect related endpoints (e.g., “indexing” routes). 
# We will later plug it into the FastAPI app:
router = APIRouter()

# We then register a POST /index endpoint on this router
@router.post("/index")
# We then define the handler function. With IndexingRequest, 
# FastAPI will parse & validate the JSON body into this object before calling the function.
def index_repo(request: IndexingRequest) -> dict:
    # We then enqueue the Celery task run_indexing_task with the provided GitHub URL..delay(...) 
    # is shorthand for apply_async(...): it serializes the args and pushes a message to your 
    # Redis broker; it returns an AsyncResult handle immediately (no waiting for the work to finish).
    task = run_indexing_task.delay(request.github_url)
    return {'task_id': task.id, 'status': 'started'}


@router.get("/repos", response_model=RepoListResponse)
async def list_indexed_repos(db: AsyncSession = Depends(get_db)) -> RepoListResponse:
    """List all indexed LinkedIn profiles"""
    # TODO: In list_indexed_repos, use the get_indexed_repos function
    # to get all the indexed repos and return an instance of RepoListResponse.
    raise NotImplemented




