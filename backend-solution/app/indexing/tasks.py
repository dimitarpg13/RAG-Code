from app.core.celery_app import celery_app
from app.indexing.indexer import Indexer
from app.indexing.github_parsing import GitHubParser   
from app.core.db import AsyncSessionLocal
from app.indexing.crud import save_indexed_repo
import asyncio
import logging

logger = logging.getLogger(__name__)

@celery_app.task
def run_indexing_task(github_url: str):
    """Celery task to run repository indexing in the background.
    
    Args:
        github_url: GitHub repository URL to index
        
    Returns:
        Result of the indexing operation
    """
    return asyncio.run(_run_indexing(github_url))

async def _run_indexing(github_url: str) -> dict[str, str | bool]:
    """Internal async function to perform repository indexing.
    
    Args:
        github_url: GitHub repository URL to index
        
    Returns:
        Dictionary with 'ok' boolean status and 'github_url' string
        
    Raises:
        Exception: If indexing fails, logs error and continues
    """
    try:
        parser = GitHubParser(github_url)
        indexer = Indexer(owner=parser.owner, repo=parser.repo, ref=parser.ref)
        data = parser.parse_repo()
        await indexer.index_data(data)
        
        # Create new DB session for background task
        async with AsyncSessionLocal() as db:
            await save_indexed_repo(db, github_url, indexer.namespace)
        
        logger.info("Indexing completed")
        return {"ok": True, "github_url": github_url}
    except Exception as e:  
        logger.error(f"Indexing failed for: {e}")
        return {"ok": False, "github_url": github_url}