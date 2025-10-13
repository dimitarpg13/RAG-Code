from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
from app.indexing.indexer import Indexer
from app.indexing.github_parsing import CodeElement
from app.chat.agents.chat_agent.state import ChatAgentState
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a routing assistant operating inside a multi-turn chat. Your sole job is to decide whether to retrieve repo context and, if so, to emit a high-signal search string. Do NOT answer the user’s question.

Inputs (JSON)
• Document samples — 3 contents from one GitHub repo (representative of the vector DB). Use ONLY to infer scope, naming patterns, and domain vocabulary.
• chat_history — ordered list of prior messages; the latest user turn(s) form the current user_query.

Output (STRICT JSON, no code fences, no extra keys)
{
  "needs_rag": true|false,
  "query_vector_db": "<3-12 keywords/phrases or null>"
}

Decision policy (recall > precision)
Use needs_rag = false only for clearly general knowledge or tiny calculations where the repo cannot add value.
Otherwise — and especially if ANY repo-specific signal appears — set needs_rag = true.

Hard “ALWAYS RAG” signals (if any appear in user_query → needs_rag = true)
• Mentions of repo entities: Class/Function/Method/Module names (CamelCase, snake_case, dotted paths like a.b.C, magic methods like __init__).
• File or path hints: *.py, *.md, “in this repo”, “where is…”, “which file defines…”.
• API specifics: signatures, args/returns, exceptions, config keys, env vars, CLI flags, endpoints, versions.
• Debug artifacts: stack traces, error messages, line numbers.
• Requests for “how to use/extend/override/instantiate <ClassName>”, “constructor params”, “example in our code”, “source of truth”.

If uncertain, default to needs_rag = true.

Non-goals
• Do NOT generate an answer, code, or explanations.
• Do NOT quote or cite. Only decide and (optionally) produce a search string.

Query crafting (when needs_rag = true)
Produce 3-12 tokens/phrases that maximize retrieval precision:
1) Include the exact symbols from user_query (ClassName, method_name, dotted paths).
2) Add the most relevant action/intent word(s): “constructor”, “signature”, “example”, “extends”, “override”, “error”, “usage”.
3) If user_query implies file type, include it literally (e.g., “.py”, “README.md”).
4) Borrow salient domain terms from Document samples (package/repo names, major frameworks like FastAPI, LangGraph, Neo4j) only if relevant to the user_query.
5) Prefer nouns/proper nouns over stopwords. No punctuation beyond dots in dotted paths and file extensions.

Examples (format to emulate)

User: “What are the constructor args of AccessGraph?”
→ {"needs_rag": true, "query_vector_db": "AccessGraph __init__ constructor parameters signature .py"}

User: “Where is PermissionRouter defined?”
→ {"needs_rag": true, "query_vector_db": "PermissionRouter class definition file path module .py"}

User: “Explain what a Python dataclass is.”
→ {"needs_rag": false, "query_vector_db": null}

User: “How do I call embed_documents in our pipeline?”
→ {"needs_rag": true, "query_vector_db": "embed_documents function usage example pipeline .py"}

Procedure
1) Extract user_query from the most recent user turn(s). Use earlier history only to resolve pronouns.
2) Check for the “ALWAYS RAG” signals. If any → needs_rag = true.
3) If none, decide: can pretrained knowledge fully answer the question without repo-specific facts? If “not clearly yes” → needs_rag = true.
4) If needs_rag = true, emit query_vector_db per “Query crafting”. If false, set query_vector_db = null.

Contract checks before output
• JSON keys exactly: needs_rag, query_vector_db.
• When needs_rag = false → query_vector_db MUST be null.
• When needs_rag = true → query_vector_db MUST be a non-empty string (3-12 tokens/phrases).
"""


class RouterDecision(BaseModel):
    """
    Decision emitted by the routing LLM.
    """
    needs_rag: bool = Field(
        ...,
        description="True if the query should be answered with RAG; False otherwise."
    )
    query_vector_db: str | None = Field(
        None,
        description="Search string for the vector DB when needs_rag is True; null otherwise."
    )


class IntentRouter:
    """
    Routes user queries to determine if RAG (Retrieval-Augmented Generation) is needed.
    
    This class analyzes the user's query and conversation history to decide whether
    the question can be answered directly or requires retrieving additional context
    from the knowledge base.
    """

    async def route(self, state: ChatAgentState) -> RouterDecision | None:
        """
        Determine routing decision for the user query.
        
        Args:
            state (ChatAgentState): Current conversation state with chat history
            
        Returns:
            RouterDecision: Contains needs_rag flag and optional search query
        """

        samples = await self.get_samples(state)
        samples_str = "\n".join([doc.model_dump_json(indent=2) for doc in samples])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Samples from Github repo  ###\n{samples_str}"},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-nano',
                temperature=0.1,
                input=messages,
                text_format=RouterDecision,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

        return response.output_parsed
    
    async def get_samples(self, state: ChatAgentState) -> list[CodeElement]:
        """
        Get sample documents from the knowledge base for routing context.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            List[str]: Sample document summaries to help with routing decision
        """
        if not state.namespace:
            raise ValueError("Chat agent state is missing the namespace for retrieval samples")
        indexer = Indexer(namespace=state.namespace)
        last_user_message = state.chat_messages[-1]['content']
        results = await indexer.search(
            last_user_message, 
            max_results=4, 
            with_filters=False, 
            with_rerank=False
        )
        return results
    
    async def __call__(self, state: ChatAgentState) -> ChatAgentState:
        """
        Execute the intent routing step.
        
        Args:
            state (ChatAgentState): Current conversation state
            
        Returns:
            ChatAgentState: Updated state with routing decision
        """
        router_decision = await self.route(state)
        state.need_rag = router_decision.needs_rag
        state.query_vector_db = router_decision.query_vector_db

        return state
    

intent_router = IntentRouter()
