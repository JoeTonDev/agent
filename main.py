from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

memory= MemorySaver()

class State(TypedDict):
  # Messages have the type "list".  The 'add_message' function
  # in the annotation defines how this state key should be updated
  # (in this case, it appends messages to the list, rather than overwiriting it)
  messages: Annotated[list, add_messages]
  
graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=5)
tools = [tool]
llm = ChatAnthropic(model="claud-3-3-sonnet-20240602")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
  return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition,)

# Any time a tool is called, we return to the chatbot to dedicde the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"],)

