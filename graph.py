#graph.py

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph

from tavily_search import tavily_search
from llm import call_llm
from prompts import SUMMARY_SYSTEM_PROMPT
from store import task_events, task_results


class AgentState(TypedDict):
    task_id: str
    input: str
    search_results: List[Dict]
    output: str


def log(task_id: str, message: str):
    task_events[task_id].append({
        "type": "log",
        "message": message
    })


def process(state: AgentState) -> AgentState:
    task_id = state["task_id"]

    log(task_id, "Starting Tavily web search")

    search_response = tavily_search(state["input"])
    results = search_response.get("results", [])

    state["search_results"] = results
    log(task_id, f"Retrieved {len(results)} search results")

    formatted_results = "\n\n".join(
        f"Title: {r.get('title')}\n"
        f"URL: {r.get('url')}\n"
        f"Content: {r.get('content')}"
        for r in results
    )

    log(task_id, "Starting LLM summarization")

    answer = call_llm(
        system=SUMMARY_SYSTEM_PROMPT,
        user=f"Question: {state['input']}\n\nSearch Results:\n{formatted_results}"
    )

    state["output"] = answer
    log(task_id, "Summarization completed")

    return state


def run_graph(task_id: str, user_input: str):
    task_events[task_id] = []

    graph = StateGraph(AgentState)
    graph.add_node("process", process)
    graph.set_entry_point("process")
    graph.set_finish_point("process")

    app = graph.compile()

    result = app.invoke({
        "task_id": task_id,
        "input": user_input
    })

    task_results[task_id] = {
        "answer": result["output"],
        "sources": [
            {"title": r.get("title"), "url": r.get("url")}
            for r in result.get("search_results", [])
        ]
    }

    task_events[task_id].append({
        "type": "completed",
        "message": "Agent execution finished"
    })
