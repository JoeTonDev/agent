from typing import Annotated
import json
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel

template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""


class State(TypedDict):
    messages: Annotated[list, add_messages]

def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_call:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"
  
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM"""
    
    objective: str
    varialbles: list[str]
    constraints: list[str]
    requirements: list[str]
    
llm = ChatAnthropic(temperature=0.5)
llm_with_tools = llm.bind_tools([PromptInstructions])

def get_messages_info(messages):
    return [SystemMessage(content=template )] + messages
  
def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage)  and m.tool_call:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs
  
def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}
 
memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)

@workflow.add_node
def add_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated",
                tool_call_id=state["messages"][-1].tool_call[0]["id"]
            )
        ]
    }
    
workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)