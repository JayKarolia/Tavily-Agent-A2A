#tavily_search.py

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

def tavily_search(query: str):
    """
    Official Tavily search
    """
    return tavily_client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False
    )
