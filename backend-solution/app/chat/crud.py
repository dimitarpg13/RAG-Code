from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
from sqlalchemy.exc import IntegrityError
from app.chat.models import Message, User, MessageType


async def get_user_id(db: AsyncSession, username: str) -> int:
    """Get or create user ID for a given username in a race-safe manner.
    
    Args:
        db: Database session
        username: Username to lookup or create
        
    Returns:
        User ID (integer)
    """
    # fast path
    uid = await db.scalar(select(User.id).where(User.username == username))
    if uid is not None:
        return uid

    # race-safe create attempt (savepoint)
    try:
        async with db.begin_nested():
            await db.execute(insert(User).values(username=username))
    except IntegrityError:
        pass  # someone else inserted it first

    # fetch id after create/race
    return await db.scalar(select(User.id).where(User.username == username))

async def save_user_message(db: AsyncSession, username: str, message: str):
    """Save a user message to the database.
    
    Args:
        db: Database session
        username: Username of the message sender
        message: Message content to save
    """
    try:
        uid = await get_user_id(db, username)
        await db.execute(insert(Message).values(
            user_id=uid,
            message=message,
            type=MessageType.USER,
        ))
        await db.commit()
    except Exception:
        await db.rollback()
        raise

async def save_assistant_message(db: AsyncSession, username: str, message: str):
    """Save an assistant response message to the database.
    
    Args:
        db: Database session
        username: Username of the user who received the response
        message: Assistant response content to save
    """
    try:
        uid = await get_user_id(db, username)
        await db.execute(insert(Message).values(
            user_id=uid,
            message=message,
            type=MessageType.ASSISTANT,
        ))
        await db.commit()
    except Exception:
        await db.rollback()
        raise

async def get_chat_history(db: AsyncSession, username: str, limit: int = 20) -> list[dict[str, str]]:
    """Retrieve chat message history for a specific user.
    
    Args:
        db: Database session
        username: Username whose chat history to retrieve
        limit: Maximum number of messages to return (default: 20)
        
    Returns:
        List of chat messages with 'role' and 'content' keys, ordered oldest-first
    """
    stmt = (
        select(Message.message, Message.type)
        .join(User, Message.user_id == User.id)
        .where(User.username == username)
        .order_by(Message.timestamp.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()  # [(message, type), ...] newest-first
    return [
        {"role": t, "content": m}
        for (m, t) in reversed(rows)       # oldest-first for chat display
    ]
