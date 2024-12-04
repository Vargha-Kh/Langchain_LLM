from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
from langchain_experimental.tools import PythonREPLTool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import GoogleSearchAPIWrapper, ArxivAPIWrapper, BingSearchAPIWrapper, \
    WikipediaAPIWrapper
from langchain.agents import Tool, ZeroShotAgent
from langchain_community.agent_toolkits.load_tools import load_huggingface_tool
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.tools import WikipediaQueryRun
import os

os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID"
os.environ["GOOGLE_API_KEY"] = "YOUR-GOOGLE-API-KEY"
os.environ["TAVILY_API_KEY"] = "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"
os.environ["BING_SUBSCRIPTION_KEY"] = "<key>"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"


# tools initialization for QA vectorstore retrievers
def retrieval_qa_tool(title, vector_db, llm):
    chain_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    return Tool(
        name=title,
        func=chain_qa.run,
        description=f"useful for when you need to answer questions about {title}. Input should be a fully formed question.",
    )


# tools initialization for vectorstore retrievers
def vectorstore_retriever_tool(title, vector_db):
    return create_retriever_tool(
        vector_db.as_retriever(),
        title,
        f"Searches and returns answers from {title} document.",
    )


def create_search_tool(engine):
    engine = engine.lower()
    if engine == 'google':
        # Google Search API tool initialization
        return Tool(
            name="Search",
            func=GoogleSearchAPIWrapper.run,
            description="useful for when you need to answer questions about current events",
        )
    elif engine == "duckduckgo":
        # DuckDuckGo Search API tool initialization
        return Tool(
            name="Search",
            func=ArxivAPIWrapper().run,
            description="useful for when you need to answer questions about papers in arxiv",
        )

    elif engine == "bing":
        # Bing Search API tool initialization
        return Tool(
            name="Search",
            func=DuckDuckGoSearchResults().run,
            description="useful for when you need to answer questions about current events",
        )

    elif engine == "arxiv":
        # Arxiv Search API tool initialization
        return Tool(
            name="Search",
            func=ArxivAPIWrapper().run,
            description="useful for when you need to answer questions about papers in arxiv",
        )

    elif engine == "wikipedia":
        # Wikipedia Search API tool initialization
        return Tool(
            name="Search",
            func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
            description="useful for when you need to answer questions about papers in arxiv",
        )
    elif engine == "tavily":
        return TavilySearchResults(max_results=1)


def huggingface_tool(task):
    huggingtool = load_huggingface_tool("lysandre/hf-model-downloads")

    print(f"{huggingtool.name}: {huggingtool.description}")
    return Tool(
        name="Search",
        func=huggingtool.run(task),
        description="useful for when you need to answer questions about papers in arxiv",
    )
