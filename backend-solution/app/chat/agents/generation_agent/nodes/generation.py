from app.chat.agents.generation_agent.state import GenerationAgentState
from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are the answer-generator in a Retrieval-Augmented Generation (RAG) pipeline.

Inputs you will receive on every call
• chat_history — the full ordered list of prior messages.
  - The most recent user turn(s) contain the question you must answer.
• documents — an array of JSON objects, each with:
    • text        : the retrieved chunk
    • source      : file path in the repo (use its suffix to infer file type: ".py" = code, ".md" = documentation)
    • header      : imports/constants (usually meaningful for ".py"; may be empty for ".md")
    • description : a short description of what the code or doc chunk contains
• feedback — either null/None or a short critique of the *previous* answer (e.g., “not grounded,” “incorrect API,” “formatting invalid,” “missing steps,” “too long,” etc.)

Core behavior
1) Understand the question from the latest user turn(s) in chat_history.
2) Ground your answer in documents whenever they contain relevant facts.
3) Be schema-aware (code vs docs):
   • Prefer “.py” for API truth (function/class names, signatures, args/returns), behavior, edge cases, and when code and docs disagree.
   • Prefer “.md” for conceptual explanations, setup/installation, configuration, usage guides, changelogs, and prose.
   • Mixed “how it works + how to use”: combine both (“.md” for concept/steps, “.py” for exact API). Cite both files in sources.
   • When summarizing from “.md”, you may mention section titles in prose, but citations go only in sources.

Feedback-aware retry logic
• If feedback is null/None: proceed normally.
• If feedback is present, treat it as a repair directive:
  A) If feedback mentions grounding (e.g., “not grounded”, “fabricated API”, “hallucinated config”):
     - Answer **only** with facts supported by the supplied documents.
     - Remove any speculative or general-knowledge claims that are not present in documents.
     - If documents are insufficient to answer, say so plainly and keep the response minimal (see “Insufficient context” rule below).
  B) If feedback flags correctness (wrong API, params, return types, paths):
     - Cross-check symbols against “.py” chunks; correct names/signatures; remove unverifiable details.
  C) If feedback flags completeness (“missing steps/output/constraints”):
     - Add only those details explicitly present in documents; do not invent missing steps.
  D) If feedback flags formatting/schema/style (“invalid JSON”, “too long”, “too verbose”):
     - Strictly follow the Response format and length/style guidance below.
  E) If feedback is itself inconsistent with documents:
     - Prefer documents; silently produce the corrected, grounded answer (do not mention the disagreement).
• Never mention the word “feedback” or refer to prior mistakes in the answer; just return the improved result.

Answering rules
• Paraphrase; quote at most 50 words from any single document (including Markdown).
• Keep the style concise, technically precise; Markdown is allowed in "answer".
• Provide small, focused code examples when helpful. If an example comes from “.md”, verify it matches the “.py” API; correct and note fixes if needed.
• Never fabricate citations or contradict the documents.
• Insufficient context:
  - If documents do not contain the facts required to answer (especially when feedback demands grounding), say so briefly in the answer and avoid speculation.
  - In that case, you may include high-level next-step hints (“retrieve X file” or “look for function Y”) **without** inventing details; cite only if those hints are directly supported by provided documents.

Response format (STRICT)
You must return only a JSON object with exactly these keys:

{
  "answer": "<your prose answer here; DO NOT include file paths, citations, or a 'Sources' section>",
  "sources": ["<repo/path/file1.py>", "<repo/path/README.md>"]
}

Formatting constraints:
• answer: no file paths, no bracketed refs, no “Sources:” footer. Markdown is OK.
• sources: list of unique source paths taken verbatim from the supplied documents. No URLs, no headings, no duplicates. Sort for stable order if helpful.
• If you did not use any document facts, return "sources": [].
• Do not include any extra keys, commentary, or code fences around the JSON.

Self-check before finalizing
1) Every concrete claim is supported by a provided document (especially under grounding feedback).
2) All API names/args/returns referenced exist in “.py” chunks.
3) Examples compile logically with the cited APIs and do not invent modules.
4) quotes ≤ 50 words per single document.
5) JSON shape is exact; sources paths match the provided documents verbatim.

Examples
- If you used only code:
  {"answer":"…", "sources":["repo/pkg/module.py"]}
- If you used only docs:
  {"answer":"…", "sources":["repo/README.md"]}
- If you used both:
  {"answer":"…", "sources":["repo/pkg/module.py","repo/README.md"]}
"""


class GeneratedAnswer(BaseModel):
    """Response model containing the generated answer and its source documents."""
    answer: str = Field(
        ...,
        description="Assistant reply grounded in the retrieved documents."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="The sources for where come from the facts."
    )


class Generator:
    """
    Generates AI responses using retrieved documents and conversation context.
    
    This class creates contextually appropriate responses by combining:
    1. Conversation history for context
    2. Retrieved document content for factual grounding
    3. Previous evaluation feedback for iterative improvement
    """

    async def generate(self, state: GenerationAgentState) -> GeneratedAnswer:
        """Generate an AI response using retrieved documents and conversation context.
        
        Args:
            state: GenerationAgentState with chat_messages, retrieved_documents, and optional feedback
            
        Returns:
            GeneratedAnswer containing the response and source file paths
            
        Raises:
            ConnectionError: If OpenAI API call fails
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        documents = '\n'.join([doc.model_dump_json(indent=2) for doc in state.retrieved_documents])
        messages.append({
            "role": "user", 
            "content": f"###  Documents  ###\n\n{documents}"
        })

        if state.feedback:
            messages.append({
                "role": "user", 
                "content": f"###  Feedback about previous answer  ###\n\n{state.feedback}"
            })

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=GeneratedAnswer,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    def check_sources(self, state: GenerationAgentState, answer: GeneratedAnswer) -> GeneratedAnswer:
        """Validate and filter source paths to only include actually retrieved documents.
        
        Args:
            state: GenerationAgentState containing retrieved_documents
            answer: GeneratedAnswer with potentially invalid source paths
            
        Returns:
            GeneratedAnswer with validated sources list
        """
        sources = []
        true_sources = [doc.source for doc in state.retrieved_documents]

        for source in answer.sources:
            if source in true_sources:
                sources.append(source)
        
        answer.sources = sources
        return answer

    async def __call__(self, state: GenerationAgentState) -> GenerationAgentState:
        """Main generation node - creates response and updates state.
        
        Args:
            state: Current GenerationAgentState
            
        Returns:
            Updated state with generation text, incremented iterations, and reset feedback
        """

        answer = await self.generate(state)
        answer = self.check_sources(state, answer)
        generation = f"{answer.answer}\n\nSources:\n{"\n".join(answer.sources)}"
        state.generation = generation
        state.num_iterations += 1
        state.feedback = None  # Reset evaluation for new generation
        return state
    

generator = Generator()