from app.chat.agents.basic_rag.state import BasicChatAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client_obs
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are the answer-generator in a Retrieval-Augmented Generation (RAG) pipeline.

Inputs you will receive on every call
• **chat_history** — the full ordered list of prior messages.
  - The most recent user turn(s) contain the question you must answer.
• **documents** — an array of JSON objects, each with:
    • `text`       : the retrieved chunk
    • `source`     : file path in the repo (use its suffix to infer file type: `.py` = code, `.md` = documentation)
    • `header`     : imports/constants (usually meaningful for `.py`; may be empty for `.md`)
    • `description`: a short description of what the code or doc chunk contains

Core behavior
1) **Understand the question** from the latest user turn(s) in *chat_history*.
2) **Ground your answer** in *documents* whenever they contain relevant facts.
3) **Be schema-aware (code vs docs)**:
   • Prefer **`.py`** for API truth (function/class names, signatures, args/returns), behavior, edge cases, and when code and docs disagree.
   • Prefer **`.md`** for conceptual explanations, setup/installation, configuration, usage guides, changelogs, and prose.
   • Mixed “how it works + how to use”: combine both (`.md` for concept/steps, `.py` for exact API). Cite both files in `sources`.
   • When summarizing from `.md`, you may mention section titles in prose, but citations go only in `sources`.

Answering rules
• Paraphrase; quote at most 50 words from any single document (including Markdown).
• Keep the style concise, technically precise; markdown allowed.
• Provide small, focused code examples when helpful. If an example comes from `.md`, verify it matches the `.py` API; correct and note fixes if needed.
• If none of the documents support the needed info:
  - answer from general knowledge **without citations**, or
  - explain that it's unavailable in the provided documents.
• **Never fabricate citations** or contradict the documents.

Response format (STRICT)
You must return **only** a JSON object with exactly these keys:

{
  "answer": "<your prose answer here; DO NOT include file paths, citations, or a 'Sources' section>",
  "sources": ["<repo/path/file1.py>", "<repo/path/README.md>"]
}

Formatting constraints:
• `answer`: no file paths, no bracketed refs, no "Sources:" footer. Markdown is OK.
• `sources`: list of unique `source` paths taken **verbatim** from the supplied `documents`. No URLs, no headings, no duplicates.
• If you did not use any document facts, return `"sources": []`.
• Do **not** include any extra keys, commentary, or code fences around the JSON.

Examples
- If you used only code:
  {"answer":"…", "sources":["repo/pkg/module.py"]}
- If you used only docs:
  {"answer":"…", "sources":["repo/README.md"]}
- If you used both:
  {"answer":"…", "sources":["repo/pkg/module.py","repo/README.md"]}
"""


class GeneratedAnswer(BaseModel):
    """Response model for RAG-generated answers with source attribution.
    
    Attributes:
        answer: Generated response grounded in retrieved documents
        sources: List of file paths cited for facts used in the answer
    """
    answer: str = Field(
        ...,
        description="Assistant reply grounded in the retrieved documents; concise, technically precise, markdown allowed."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of file paths (`source`) cited for facts used in the answer; include each path exactly once."
    )


class Generator:
    """Answer generator component for RAG chat agent.
    
    Generates contextual responses by combining chat history with retrieved
    documents using OpenAI's language model and structured output parsing.
    """

    async def generate(self, state: BasicChatAgentState) -> GeneratedAnswer:
        """Generate a grounded answer using chat history and retrieved documents.
        
        Args:
            state: Current chat agent state with messages and retrieved documents
            
        Returns:
            GeneratedAnswer with response text and source citations
            
        Raises:
            ConnectionError: If OpenAI API call fails
            
        Note:
            Uses last 10 chat messages for context and formats retrieved documents
            as JSON for the language model to process and cite appropriately.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        documents = '\n'.join([doc.model_dump_json(indent=2, exclude_none=True) for doc in state.retrieved_documents])
        messages.append({
            "role": "user", 
            "content": f"###  Documents  ###\n\n{documents}"
        })

        try:
            response = await async_openai_client_obs.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=GeneratedAnswer,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def __call__(self, state: BasicChatAgentState) -> BasicChatAgentState:
        """Execute generation step and update state with final response.
        
        Args:
            state: Current chat agent state
            
        Returns:
            Updated state with generation field containing formatted answer and sources
            
        Note:
            Formats the generated answer with sources list for display to user.
            This method makes the Generator callable as a node in the chat agent graph.
        """
        answer = await self.generate(state)
        generation = f"{answer.answer}\n\nSources:\n{"\n".join(answer.sources)}"
        state.generation = generation
        return state


# Global generator instance for use in chat agent graph
generator = Generator()