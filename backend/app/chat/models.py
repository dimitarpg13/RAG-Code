from sqlalchemy import ForeignKey, String, DateTime, func, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from app.core.db import Base
import enum


class MessageType(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class User(Base):
    """
    User model representing registered users in the system.
    
    Attributes:
        id (int): Primary key identifier
        username (str): Unique username for the user
        messages (List[Message]): All messages associated with this user
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    messages = relationship("Message", back_populates="user")

class Message(Base):
    """
    Message model representing chat messages in conversations.
    
    Attributes:
        id (int): Primary key identifier
        user_id (int): Foreign key reference to the user who sent the message
        message (str): The actual message content
        type (str): Message type ('user' or 'assistant')
        timestamp (datetime): When the message was created
        user (User): The user who sent this message
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete='CASCADE'), index=True)
    message: Mapped[str] = mapped_column(String(10000))
    type: Mapped[MessageType] = mapped_column(Enum(MessageType))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user = relationship("User", back_populates="messages")