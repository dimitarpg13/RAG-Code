from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.chat.agents.chat_agent.agent import chat_agent
from app.chat.agents.chat_agent.state import ChatAgentState
from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.crud import get_chat_history, save_user_message, save_assistant_message
from app.core.db import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    # TODO implement the chat function:
    # 1. When we receive the request, we need to save the message
    # 2. We then retrieve the chat history
    # 3. We then create the initial state for ChatAgentState.
    # 4. We then invoke the chat_agent.ainvoke(...).
    # 5. We then cast the final state into a ChatAgentState object.
    # 6. We then save the generated message
    # 7. And finally, we return the response.

    try:
        # Save user message first
    
        # Retrieve chat history from database
        
        # Initialize state
        
        # Run the agent
        
        # Cast result to ChatAgentState
        
        # Save assistant response to database
        
        raise NotImplemented
        
    except Exception as e:
        logger.error(f"Chat agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))