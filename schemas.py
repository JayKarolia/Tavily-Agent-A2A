#schemas.py

from pydantic import BaseModel
from typing import Optional, List, Dict

class InvokeRequest(BaseModel):
    input: str

class InvokeResponse(BaseModel):
    task_id: str
    status: str

class Event(BaseModel):
    type: str
    message: str

class ResultResponse(BaseModel):
    status: str
    output: Optional[Dict] = None
