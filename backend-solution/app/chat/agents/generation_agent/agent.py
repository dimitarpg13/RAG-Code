from langgraph.graph import END, StateGraph, START
from app.chat.agents.generation_agent.state import GenerationAgentState
from app.chat.agents.generation_agent.nodes.generation import generator
from app.chat.agents.generation_agent.nodes.evaluation import evaluator
from dataclasses import dataclass


@dataclass(frozen=True)
class Nodes:
    GENERATOR = "generator"
    EVALUATOR = 'evaluator'

builder = StateGraph(GenerationAgentState)

builder.add_node(Nodes.GENERATOR, generator)
builder.add_node(Nodes.EVALUATOR, evaluator)

builder.add_edge(START, Nodes.GENERATOR)
builder.add_edge(Nodes.GENERATOR, Nodes.EVALUATOR)

def generation_evaluation_router(state: GenerationAgentState):
    """Route based on response quality evaluation results."""
    if state.is_grounded and state.is_valid:
        return END
    elif state.num_iterations <= 3:
        return Nodes.GENERATOR
    else: 
        return END

builder.add_conditional_edges(
    Nodes.EVALUATOR,
    generation_evaluation_router,
    {
        Nodes.GENERATOR: Nodes.GENERATOR,
        END: END
    }

)

generator_agent = builder.compile()