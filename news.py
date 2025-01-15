import os
from typing import TypedDict, Annotated, List
from langgraph.graph import Graph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables.graph import MermaidDrawMethod
from datetime import datetime
import re
from getpass import getpass
from dotenv import load_dotenv
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
from IPython.display import display, Image as IPImage
import asyncio


