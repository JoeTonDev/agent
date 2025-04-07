from typing import Annotated, List, Tuple
import json
import getpass
import os
import operator
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union, Literal

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter {var}: ")
        
_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use
prompt = hub.pull('ih/ih-react-agent-executor')
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    
class Plan(BaseModel):
    """The plan to execute"""
    
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
planner_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatAnthropic(
    model="claude-3-5-sonnet-20240620", temperature=0.5
    
).with_structured_output(Plan)

planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)

# The plan is a list of steps to follow
class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatAnthropic(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
      
workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)
# Add the execution step
workflow.add_node("agent", execute_step)
# Add a replan node
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
# From plan we go to agent
workflow.add_edge("planner", "agent")
# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    {"agent", END}
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile 


