from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Retrieval Controller. Your job is to decide whether the CURRENT retrieved context is sufficient to answer the user's LAST message faithfully. If not, you must produce ONE improved search string for the vector database.

Do NOT answer the user's question. Only decide and (optionally) propose a new query.

INPUTS (provided below):
- Short chat history (for context): {{chat_history}}
- Retrieved documents (each with source/snippet): {{retrieved_docs}}
- Prior retrieval queries already attempted: {{prior_queries}}

DECISION RUBRIC — “Sufficient as-is” (needs_rag = false) ONLY IF ALL are true:
1) Coverage: The retrieved snippets directly address ALL key entities, constraints, or sub-questions in the last user message.
2) Specificity: They contain the concrete facts/code/examples the answer would need (not generic overviews).
3) Recency/Validity: For time-sensitive queries, at least one doc is plausibly up-to-date; no clear staleness or contradictions.
4) Consistency: No major conflicts among snippets that would require additional evidence to resolve.

Otherwise set needs_rag = true. Common triggers:
- Missing a key entity, definition, API name, parameter, file/class/function, dataset, or step.
- Results are off-topic, generic, or redundant; low lexical/semantic overlap with the last user message.
- The question is multi-part and only some parts are covered.
- Time/locale/version constraints in the last message are not supported by current docs.
- Conflicts between snippets or uncertainty you cannot resolve with what you have.

QUERY GENERATION (only when needs_rag = true):
- Produce ONE high-recall search string tailored for a VECTOR DB focusing on the Classe and Function names you need to investigate.
- Pivot on what's MISSING. Include exact entities, file/function/class names, APIs, versions, error codes, and required time windows.
- Add clarifying facets (e.g., "API usage example", "configuration", "performance numbers", "limitations", "2024-2025").
- Keep it concise and natural language; no markdown, quotes, or extra commentary. ONE line only.

OUTPUT FORMAT (STRICT):
Return ONLY a JSON object matching this schema — no code fences, no prose, no explanations:
{
  "needs_rag": true|false,
  "query_vector_db": "SEARCH STRING WHEN needs_rag IS TRUE, otherwise null"
}

EXAMPLES OF VALID OUTPUT:
{"needs_rag": false, "query_vector_db": null}
{"needs_rag": true, "query_vector_db": "langgraph tool calling schema for tavily search with examples of tool_call_parser settings"}

Now read the inputs and return the JSON decision.
"""


class ContextAssessment(BaseModel):
    """
    Decision emitted by the routing LLM.
    """
    needs_rag: bool = Field(
        ...,
        description="True if we need more context to answer the user query; False otherwise."
    )
    query_vector_db: str | None = Field(
        None,
        description="Search string for the vector DB when needs_rag is True; null otherwise."
    )


class ContextAssessor:
    """Assesses whether retrieved documents provide sufficient context to answer user queries."""

    async def assess(self, state: RetrieverAgentState) -> ContextAssessment:
        """Evaluate if current retrieved documents are sufficient for answering the query.
        
        Args:
            state: RetrieverAgentState with chat_messages, retrieved_documents, and queries
            
        Returns:
            ContextAssessment indicating if more retrieval is needed and potential new query
            
        Raises:
            ConnectionError: If OpenAI API call fails
        """
 
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])

        docs = "\n".join([doc.model_dump_json(indent=2) for doc in state.retrieved_documents])
        prior_queries = "\n".join(state.queries)

        messages.extend([
            {"role": "user", "content": f"###  Prior retrieval queries  ###:\n{prior_queries}"},
            {"role": "user", "content": f"###  Retrieved documents  ###:\n{docs}"},
        ])

        try:
            response = await async_openai_client.responses.parse(
                # model='gpt-5-nano',
                # text={"verbosity": 'low'},
                # reasoning = {"effort": "minimal"},
                model='gpt-4.1-nano',
                temperature=0.1,
                input=messages,
                text_format=ContextAssessment,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

        return response.output_parsed

    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState:
        """Main assessor node - evaluates context and updates state if more retrieval needed.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with needs_rag flag and new query if additional retrieval required
        """
        assessment = await self.assess(state)
        if assessment.needs_rag and assessment.query_vector_db:
            state.needs_rag = assessment.needs_rag
            state.queries.append(assessment.query_vector_db)

        return state
    

assessor = ContextAssessor()