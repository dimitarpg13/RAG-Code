from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.indexing.models import IndexedRepo


async def save_indexed_repo(
    session: AsyncSession,
    github_url: str,
    namespace: str
) -> IndexedRepo:
    """Save or retrieve an indexed repository record.
    
    Args:
        session: Database session
        github_url: GitHub repository URL
        namespace: Repository namespace identifier
        
    Returns:
        IndexedRepo object, either existing or newly created
    """

    # TODO: Create a statement that will select all the rows of the IndexedRepo 
    # table where the column github_url is equal to the github_url argument.
    statement = None
    # The following line will execute that statement
    result = await session.execute(statement)
    repo = result.scalar_one_or_none()
    
    if not repo:
        # TODO:  if repo is None: 
        # - create an instance IndexedRepo
        # - Use the add function to add the instance to the session.   

        # Flushes the pending INSERT and commits the transaction.
        await session.commit()
        # Immediately re-loads the row from the DB so repo has all 
        # server-generated fields populated (e.g., autoincrement id, default timestamps).
        await session.refresh(repo)

    return repo


async def get_indexed_repos(
        session: AsyncSession
    ) -> list[IndexedRepo]:
    """Retrieve all indexed repositories ordered by most recent.
    
    Args:
        session: Database session
        
    Returns:
        List of IndexedRepo objects ordered by indexed_at descending
    """
    # TODO: Return all the indexed repos in descending order of indexed_at. 
    # The select statement will select all the rows from the table, and the order_by function will order them.
    statement = None
    result = await session.execute(statement)
    return result.scalars().all()