# Python built in libraries
from typing import Annotated, List # used to set additional metadata for a variable  
from typing_extensions import TypedDict # a type that allows you to define dictionaries with specific key-value types
import os

# Langchain / Langraphpip 
from langgraph.graph import (StateGraph, # data structure which represents the current snapshot of an application 
                             START # type of node (a python function which has some kind of logic) which takes user input and sends it into the graph 
                            )
from langgraph.graph.message import add_messages # appends messages to the end of the attribute it was assigned to
from langgraph.prebuilt import (ToolNode, # a pre-built component and node whichs runs the tools called in the last AIMessage 
                                tools_condition # a pre-built component and node which uses the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
                                )
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, BaseMessage

# Tools
from tools import get_token_balance, lend_crypto, borrow_crypto, set_private_key
from web3 import Web3

# User Interface
import streamlit as st

# Initialize Web3
web3 = Web3(Web3.HTTPProvider(os.getenv("RPC_URL")))

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", key="chatbot_api_key", type="password")
    "[Get an Groq API key](https://console.groq.com/keys)"
    
    private_key = st.text_input("Private Key", key="private_key", type="password")
    "[Create a private key with Rabby Wallet](https://rabby.io/)"

    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AAVE DeFi assistant. I can help you check token balances, lend, and borrow crypto. How can I assist you today?"}]
        st.rerun()

    "[![View the source code](https://badgen.net/static/Github/Repository/black?icon=github)](https://github.com/jondoescoding/awesome-ai-agents/tree/main/ai_agents/aave_agent)"

    
if not groq_api_key:
    st.warning(body="API key is not set. Please set the Groq API key.")
    st.stop()
if not private_key:
    st.warning(body="Private key is not set. Please set your wallet private key.")
    st.stop()

# Set the private key in tools.py
set_private_key(private_key)

# Create account from private key
account = web3.eth.account.from_key(private_key)
user_address = account.address

# System prompt with token information and user's address
SYSTEM_PROMPT = f"""You are an AAVE DeFi assistant that helps users interact with the AAVE protocol on Ethereum.
You are connected to the wallet address: {user_address}

You have access to the following tokens and their addresses:

- USDC (USD Coin): 0x94a9D9AC8a22534E3FaCa9F4e7F2E2cf85d5E4C8
- DAI (Dai Stablecoin): 0xFF34B3d4Aee8ddCd6F9AFFFB6Fe49bD371b8a357 
- WBTC (Wrapped Bitcoin): 0x29f2D40B0605204364af54EC677bD022dA425d03 
- USDT (Tether USD): 0xaA8E23Fb1079EA71e0a56F48a2aA51851D8433D0

You can help users:
1. Check their token balances of ONLY the above contracts. Let the user know what tokens are available.
2. Lend their tokens to earn interest
3. Borrow tokens against their collateral

Always use the exact token addresses provided above when helping users interact with the protocol."""

llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

# Initialize LLM and tools
tools = [get_token_balance, lend_crypto, borrow_crypto]
llm_with_tools = llm.bind_tools(tools=tools)

# State Management
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
  
# Graph builder setup
graph_builder = StateGraph(State)

# Chatbot function
def chatbot(state: State):
    # Get all messages including history
    messages = state["messages"]
    
    # Add system message if it's not already in the messages
    if not any(isinstance(message, SystemMessage) for message in messages):
        messages = (SystemMessage(content=SYSTEM_PROMPT)) + messages
        
    # Get response from LLM with full conversation history
    response = llm_with_tools.invoke(messages)
    
    return {"messagers": [response]}
    
    
# Node configuration
tools_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("chatbot", chatbot)

# Edge configuration
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_condition_edge("chatbot", tools_condition)

# Complete the graph
graph = graph_builder.compile()

####
# UI - Streamlit Chat Interface
####

st.title("AAVE DeFi Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your AAVE DeFi assistant. I can help you check token balances, lend, and borrow crypto. How can I assist you today?"}]
    
# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
    
# Handle user input
if prompt := st.chat_input():
    if not groq_api_key:
        st.info("Please set the Groq API key.")
        st.stop()
        
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message(role="user").write(prompt)
    
    # Convert chat history to Langchain format
    from langchain_core.messages import AIMessage, HumanMessage
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
            
    # Get AI response with a loading spinner
    with st.spinner("Thinking..."):
        response = graph.invoke({"messages": history})
        msg = response["messages"][-1].content
        
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message(role="assistant").write(msg)