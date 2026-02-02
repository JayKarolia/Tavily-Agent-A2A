from fastapi import FastAPI, BackgroundTasks
from uuid import uuid4

from schemas import InvokeRequest, InvokeResponse, ResultResponse, Event
from graph import run_graph
from store import task_results, task_events

app = FastAPI(
    title="Tavily LangGraph External Agent (A2A + JSON-RPC)",
    version="2.1.0"
)



@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(
    request: InvokeRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid4())

    background_tasks.add_task(
        run_graph,
        task_id,
        request.input
    )

    return InvokeResponse(
        task_id=task_id,
        status="started"
    )


@app.get("/events/{task_id}", response_model=list[Event])
async def get_events(task_id: str):
    return task_events.get(task_id, [])


@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    if task_id in task_results:
        return ResultResponse(
            status="completed",
            output=task_results[task_id]
        )
    return ResultResponse(status="running")



@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "tavily-search-agent",
        "description": "Web search and summarization agent using Tavily and LangGraph",
        "version": "1.0.0",
        "protocol": "a2a",
        "skills": [
            {
                "name": "web_search",
                "description": "Search the web and summarize results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "sources": {"type": "array"}
                    }
                }
            }
        ],
        "endpoints": {
            "invoke": "/a2a/message/send",
            "status": "/a2a/tasks/get"
        }
    }




def jsonrpc_response(id, result=None, error=None):
    response = {
        "jsonrpc": "2.0",
        "id": id
    }
    if error:
        response["error"] = error
    else:
        response["result"] = result
    return response




@app.post("/a2a/message/send")
async def a2a_message_send(payload: dict, background_tasks: BackgroundTasks):
    """
    JSON-RPC compliant A2A message send
    """
    try:
        if payload.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")

        request_id = payload["id"]
        method = payload["method"]
        params = payload.get("params", {})

        if method != "message/send":
            raise ValueError("Unsupported method")

        query = params["content"]["query"]

        task_id = str(uuid4())

        background_tasks.add_task(
            run_graph,
            task_id,
            query
        )

        return jsonrpc_response(
            request_id,
            result={
                "task_id": task_id,
                "status": "accepted"
            }
        )

    except Exception as e:
        return jsonrpc_response(
            payload.get("id"),
            error={
                "code": -32600,
                "message": str(e)
            }
        )


@app.post("/a2a/tasks/get")
async def a2a_tasks_get(payload: dict):
    """
    JSON-RPC compliant task polling
    """
    try:
        if payload.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version")

        request_id = payload["id"]
        params = payload["params"]
        task_id = params["task_id"]

        if task_id in task_results:
            return jsonrpc_response(
                request_id,
                result={
                    "task_id": task_id,
                    "status": "completed",
                    "result": task_results[task_id]
                }
            )

        if task_id in task_events:
            return jsonrpc_response(
                request_id,
                result={
                    "task_id": task_id,
                    "status": "running"
                }
            )

        return jsonrpc_response(
            request_id,
            result={
                "task_id": task_id,
                "status": "unknown"
            }
        )

    except Exception as e:
        return jsonrpc_response(
            payload.get("id"),
            error={
                "code": -32602,
                "message": str(e)
            }
        )



@app.get("/health")
def health():
    return {"status": "ok"}
