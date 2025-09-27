from langgraph.graph import END, StateGraph, START
from app.chat.agents.generation_agent.state import GenerationAgentState
from app.chat.agents.generation_agent.nodes.generation import generator
from app.chat.agents.generation_agent.nodes.evaluation import evaluator
from dataclasses import dataclass


@dataclass(frozen=True)
class Nodes:
    GENERATOR = "generator"
    EVALUATOR = 'evaluator'


# TODO: Instantiate StateGraph with GenerationAgentState.
builder = None
# TODO: Add the generator and evaluator nodes to the graph.
# TODO: Add an edge from START to the generator and from the generator to the evaluator.
 
def generation_evaluation_router(state: GenerationAgentState) -> str:
    # TODO: Implement generation_evaluation_router for a conditional edge. If is_grounded and 
    # is_valid, end the pipeline, otherwise if num_iterations <= 3, retry the generation.
    """Route based on response quality evaluation results."""
    raise NotImplemented


# TODO: Add a conditional edge from the evaluator with generation_evaluation_router.

generator_agent = builder.compile()