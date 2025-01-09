from typing import Annotated
import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    ask_human: bool
    
class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str
    

tool = TavilySearchResults(max_results=2)
tools = [tool] 
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

def chatbot(state: State):
  response = llm_with_tools.invoke(state["messages"])
  ask_human = False
  if (
    response.tools_calls
    and response.tool_calls[0]["name"] == RequestAssistance.__name__
  ):
    ask_human = True
  return {"messages": [response], "ask_human": ask_human}
  
def select_next_node(state: State):
  if state["ask_human"]:
    return "human"
  # Otherwise, return the default node
  return tools_condition(state)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node(tools=[tool]))

def create_response(response: str, ai_message: AIMessage):
  return ToolMessage(
    content=response,
    tool_call_id=ai_message.tool_call[0]["id"],
  )
  
def human_node(state: State):
  new_messge = []
  if not isinstance(state["messages"][-1], ToolMessage):
    # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messge.append(
          create_response("No response from human", state["messages"][-1])
        )
  return {
    # Append the new messages
    "messages": new_messge,
    "ask_human": False,}
  
graph_builder.add_node("human", human_node)

def select_next_node(state: State):
  if state["ask_human"]:
    return "human"
  # Otherwise, return the default node
  return tools_condition(state)

graph_builder.add_conditional_edges(
  "chatbot",
  tools_condition,
  {"human": "human", "tools": "tools", "__end__": "__end__"},
)
  
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(
  checkpointer=memory,
  interupt_before=["human"],)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

 