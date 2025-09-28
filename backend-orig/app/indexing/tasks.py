from app.core.celery_app import celery_app
from app.indexing.indexer import Indexer
from app.core.db import AsyncSessionLocal
from app.indexing.github_parsing import GitHubParser  
from app.indexing.crud import save_indexed_repo
import asyncio
import logging

logger = logging.getLogger(__name__)

@celery_app.task
def run_indexing_task(github_url: str) -> dict[str, str | bool]:
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
        
    Raises:
        Exception: If indexing fails, logs error and continues
    """
    # TODO: Implement _run_indexing:
    try:
        # TODO: Instantiate GitHubParser with github_url
        # TODO: Instantiate Indexer with the parser.
        # TODO: Call the parse_repo() function on the parser
        # TODO: Run the index_data() function on the indexer.
        
        # Create new DB session for background task
        # AsyncSessionLocal is just a factory that gives you new SQLAlchemy AsyncSession objects, 
        # each bound to your database engine. The “Local” in the name means “local to this 
        # process/thread as a factory,” not “local machine.” Each AsyncSessionLocal() call gives you a fresh, 
        # lightweight session object that opens DB connections on demand and cleans them up when the async with block exits.
        async with AsyncSessionLocal() as db:
            # TODO: Save the indexed repo with the save_indexed_repo function
            pass
        
        logger.info("Indexing completed")
        return {"ok": True, "github_url": github_url}
    except Exception as e:  
        logger.error(f"Indexing failed for: {e}")
        return {"ok": False, "github_url": github_url}