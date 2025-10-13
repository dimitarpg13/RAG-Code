from app.chat.agents.chat_agent.state import ChatAgentState
from app.chat.agents.generation_agent.agent import generator_agent
from app.chat.agents.generation_agent.state import GenerationAgentState


class Generator:

    async def subagent(self, state: ChatAgentState) -> ChatAgentState:
        initial_generation_state = GenerationAgentState(
            chat_messages=state.chat_messages,
            retrieved_documents=state.retrieved_documents,
        )
        response = await generator_agent.ainvoke(
            initial_generation_state.model_dump()
        )
        final_generation_state = GenerationAgentState(**response)
        state.generation = final_generation_state.generation

        return state    
    
generator = Generator()