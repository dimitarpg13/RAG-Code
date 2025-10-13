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

    statement = select(IndexedRepo).where(IndexedRepo.github_url == github_url)
    result = await session.execute(statement)
    repo = result.scalar_one_or_none()
    
    if not repo:

        # Create new profile
        repo = IndexedRepo(
            github_url=github_url,
            namespace=namespace
        )
        session.add(repo)
    
        await session.commit()
        await session.refresh(repo)

    return repo


async def get_indexed_repo_by_url(
    session: AsyncSession,
    github_url: str
) -> IndexedRepo | None:
    """Return the indexed repo record matching the provided GitHub URL."""
    statement = select(IndexedRepo).where(IndexedRepo.github_url == github_url)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def get_indexed_repos(
        session: AsyncSession
    ) -> list[IndexedRepo]:
    """Retrieve all indexed repositories ordered by most recent.
    
    Args:
        session: Database session
        
    Returns:
        List of IndexedRepo objects ordered by indexed_at descending
    """
    stmt = select(IndexedRepo).order_by(
        IndexedRepo.indexed_at.desc()
    )
    result = await session.execute(stmt)
    return result.scalars().all()
