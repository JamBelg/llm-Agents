from dotenv import load_dotenv
import os
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools import TavilySearchResults
import matplotlib
import pandas
import json
import lxml
import datetime
from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentExecutor, Tool
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate


def define_env():
    # Load env file
    load_dotenv()
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

def create_agent():
    chat = ChatCohere(model="command-r-plus", temperature=0.2)
    return chat

def web_search():
    """First tool: web search using TavilySearch"""
    internet_search = TavilySearchResults(
        max_results=10,
        include_answer=True,
        include_raw_content=True,
        include_images=False
        )
    internet_search.name = "internet_search"
    internet_search.description = """
    Returns a list of relevant document snippets for a textual query retrieved from the internet."""
    return internet_search

def python_tool():
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
        func=python_repl.run,
    )
    repl_tool.name = "python_interpreter"
    return repl_tool

def main():
    define_env()
    chat = create_agent()
    tool1 = web_search()
    tool2 = python_tool()
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_template("{input}")

    # Create the ReAct agent
    agent = create_cohere_react_agent(
        llm=chat,
        tools=[tool1,tool2],
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[tool1, tool2],
        verbose=True
    )
    try:
        result = agent_executor.invoke({
            "input": "Find the gold price history in us dollar by once for the last month day by day, transform it to a table of numerics and plot it",
        })
    except AttributeError as e:
        print(f"Error: {e}")
    print(result)

if __name__ == "__main__":
    main()