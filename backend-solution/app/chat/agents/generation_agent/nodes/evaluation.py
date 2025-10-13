from app.chat.agents.generation_agent.state import GenerationAgentState
from app.core.clients import async_openai_client
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an evaluation assistant for a Retrieval-Augmented Generation (RAG) system over a GitHub repository.

Inputs (JSON)
• chat_history — the full ordered list of prior messages; the latest user turn(s) contain the question.
• documents — the exact set of retrieved repo chunks the generator saw; each item has:
    • text         : chunk content
    • source       : repo-relative path (e.g., "pkg/mod.py"; ".py" = code, ".md" = docs)
    • header       : imports/constants (often meaningful for ".py"; may be empty for ".md")
    • description  : short description of the chunk
• generation — the generator's reply as a string containing with keys:
    {
      "answer": "<model prose; markdown allowed; no file paths>",
      "sources": ["<repo/path1.ext>", "<repo/path2.ext>"]
    }

Your goals
1) Hallucination / grounding check (repo-aware)
   • Parse answer_json. If parsing fails or required keys are missing → not grounded.
   • If sources is non-empty: every explicit factual claim in "answer" that references entities present in the provided documents
     (e.g., function/class names, signatures, params/returns, config keys, env vars, CLI flags, endpoints, file names/paths,
     algorithms/protocols, version constraints, side effects, I/O) must be supported by at least one of the provided documents.
   • If sources is empty: treat the reply as general knowledge. It is acceptable as long as it does NOT contradict any provided document.
     (Do NOT require citations when sources == [].)
   • Code-vs-docs precedence:
       - Prefer ".py" chunks for API truth (names, signatures, behavior). If ".md" disagrees with ".py", ".py" wins.
       - ".md" is preferred for conceptual usage, setup, configuration, and workflows.
     When the answer asserts API facts that are only present in ".md" and conflict with ".py", mark as unsupported.
   • Quotes: the answer must quote ≤ 50 words from any single document. Longer verbatim copy is an issue.
   • The "sources" array must contain only paths that appear as "source" in documents, be unique, and exclude URLs/headings.

2) Answer validity (task satisfaction)
   • The answer must be relevant to the latest user question in chat_history, technically correct, and complete given the supplied documents.
   • Small code examples are welcome only if they match the actual ".py" APIs (names, params, returns) found in documents.
   • If the documents are insufficient, a brief acknowledgment of missing context is valid; the answer must avoid speculation and may give high-level next steps only if supported by documents.

3) Format contract compliance
   • The generator must return valid JSON with exactly the keys "answer" and "sources".
   • "answer": no file paths or bracketed citations; markdown allowed.
   • "sources": zero or more unique repo paths copied verbatim from the provided documents’ source fields.

Decisions to output (STRICT)
Return only this JSON object:

{
  "is_grounded": <true|false>,
  "is_valid": <true|false>,
  "feedback": "<actionable suggestion 1>, <actionable suggestion 2>, ..."  // or null when both booleans are true
}

Decision policy
• is_grounded = true iff:
   - answer_json parses AND
   - sources ⊆ {documents.source} (when non-empty) AND
   - no explicit factual claims conflict with documents AND
   - when sources ≠ [], all asserted doc-based claims are supported by at least one provided document AND
   - quotes per single document ≤ 50 words.
  Otherwise false.
• is_valid = true iff:
   - the answer addresses the latest user question, is technically correct given documents, respects code vs docs precedence,
     provides accurate examples (when present), and adheres to the format contract.
  Otherwise false.

Feedback construction (single merged list)
• Write concrete, next-step suggestions the generator can follow on the next round. Merge grounding and validity issues into ONE list.
• Use patterns like:
   - Grounding: "Remove claim '<X>' or cite '<path.py>' that supports it."
   - Precedence: "Verify API names/params against '<path.py>'; update to match code."
   - Completeness: "Add the missing steps for <task> described in '<path.md>' section '<header>'."
   - Examples: "Update example to use '<func(arg=...)>' per '<path.py>' signature."
   - Quotes: "Paraphrase long quote from '<path.md>' to under 50 words."
   - Format: "Return valid JSON with keys 'answer' and 'sources' only."
• If both is_grounded and is_valid are true, set feedback to null.

Edge cases
• If sources contains a path not in documents: mark is_grounded=false; suggest replacing with a valid path.
• If answer contradicts ".py" APIs when ".md" differs: mark unsupported; suggest aligning with ".py".
• If answer uses file paths or inline citations in prose: keep or flip is_valid to false (format breach) and suggest removing paths from "answer".
"""

class EvaluationResult(BaseModel):

    is_grounded: bool = Field(..., description="True if all claims are supported by the documents.")
    is_valid: bool = Field(..., description="True if the answer fully addresses the user's question.")
    feedback: str | None = Field(
        None,
        description="Merged list of actionable suggestions when issues exist; null if no issues."
    )


class GenerationEvaluator:
    """
    Evaluates generated responses for factual accuracy and completeness.
    
    This class performs quality control on AI-generated responses by checking:
    1. Hallucination detection - ensuring claims are supported by retrieved documents
    2. Answer validity - verifying the response addresses the user's question properly
    """

    async def evaluate(self, state: GenerationAgentState) -> EvaluationResult:
        """
        Evaluate the quality of a generated response.
        
        Args:
            state (ChatAgentState): Current conversation state with generated response
            
        Returns:
            EvaluationResult: Evaluation with grounding/validity flags and feedback
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

        messages.append({
            "role": "assistant", 
            "content": f"###Generation ###: {state.generation}"
        })

        try:
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-mini',
                input=messages,
                temperature=0.1,
                text_format=EvaluationResult,
            )
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError("Something wrong with Openai: {e}")

        return response.output_parsed
    
    async def __call__(self, state: GenerationAgentState) -> GenerationAgentState:
        """Main evaluation node - evaluates generation and updates state flags.
        
        Args:
            state: Current GenerationAgentState with generation to evaluate
            
        Returns:
            Updated state with evaluation results and cleared generation if invalid
        """
        evaluation = await self.evaluate(state)
        state.is_grounded = evaluation.is_grounded
        state.is_valid = evaluation.is_valid
        state.feedback = evaluation.feedback
        if not (evaluation.is_grounded and evaluation.is_valid):
            state.generation = None

        return state
    

evaluator = GenerationEvaluator()