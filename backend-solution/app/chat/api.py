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
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):

    try:
        # Save user message first
        await save_user_message(db, request.username, request.message)
        
        # Retrieve chat history from database
        chat_messages = await get_chat_history(db, request.username)
        # Initialize state with the indexed namespace linked to this URL
        initial_state = ChatAgentState(
            chat_messages=chat_messages,
            namespace=request.namespace
        )
        
        # Run the agent
        result = await chat_agent.ainvoke(initial_state, debug=True)
        
        # Cast result to ChatAgentState
        final_state = ChatAgentState(**result)
        
        # Save assistant response to database
        response_text = final_state.generation or "I'm sorry, I couldn't generate a response."
        await save_assistant_message(db, request.username, response_text)
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Chat agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
