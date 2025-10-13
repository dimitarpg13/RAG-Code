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

router = APIRouter()


@router.post("/index")
async def index_repo(request: IndexingRequest):
    task = run_indexing_task.delay(request.github_url)
    return {'task_id': task.id, 'status': 'started'}


@router.get("/repos", response_model=RepoListResponse)
async def list_indexed_repos(db: AsyncSession = Depends(get_db)):
    """List all indexed LinkedIn profiles"""
    repos = await get_indexed_repos(db)
    return RepoListResponse(repos=repos)
    


