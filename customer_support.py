from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import AnyMessage, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
        
    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tools_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
                
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

builder = StateGraph(State)

builder.add_node("assistant, Assistant(runnable=llm)")
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edge("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)