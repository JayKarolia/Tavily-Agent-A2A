#agent.py

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSequence

# Tavily tool
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced"
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Prompt
prompt = PromptTemplate(
    input_variables=["question", "search_results"],
    template="""
You are a web research assistant.

Use the search results below to answer the question.
- Be factual
- Be concise
- Do not hallucinate
- If information is missing, say so clearly

Question:
{question}

Search Results:
{search_results}

Provide a clear summarized answer.
"""
)

# Chain
chain = RunnableSequence(
    {
        "search_results": lambda x: search_tool.invoke(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)
