from app.chat.agents.generation_agent.state import GenerationAgentState
from app.core.clients import async_openai_client
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


# TODO: Implement your system prompt to generate the evaluation result.
SYSTEM_PROMPT = None

class EvaluationResult(BaseModel):
    # TODO: Implement EvaluationResult. We need the following attributes:
    # - is_grounded: a boolean to assess if the generation is grounded in the retrieved data
    # - is_valid: a boolean to assess if the generation correctly addresses the user's question
    # - feedback: a string to capture the necessary feedback if is_grounded or is_valid is true. 
    pass


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
        # TODO: Implement the evaluate function. Use the system prompt, the chat history, 
        # the retrieved documents, EvaluationResult, and the generation to assess it.
        raise NotImplemented
    
    async def __call__(self, state: GenerationAgentState) -> GenerationAgentState:
        """Main evaluation node - evaluates generation and updates state flags.
        
        Args:
            state: Current GenerationAgentState with generation to evaluate
            
        Returns:
            Updated state with evaluation results and cleared generation if invalid
        """
        # TODO: In __call__, 
        # - use the evaluate function to generate the evaluation.
        # - update the state's is_valid, is_grounded and feedback attributes
        # - if is_valid or is_grounded are not true, reset the generation to None.
        return state
    

evaluator = GenerationEvaluator()