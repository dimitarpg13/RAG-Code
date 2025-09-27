from app.core.db import Base
from sqlalchemy import String, DateTime, func
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column


class IndexedRepo(Base):
    """Database model for tracking indexed GitHub repositories."""

    __tablename__ = "indexed_repos"

    id: Mapped[int] = mapped_column(
        primary_key=True, 
    )
    github_url: Mapped[str] = mapped_column(
        String(2048), unique=True
    )
    namespace: Mapped[str] = mapped_column(
        String(2048), unique=True
    )
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )