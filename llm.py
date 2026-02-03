#llm.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL = os.getenv("OPENROUTER_LLM_MODEL")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

def call_llm(system: str, user: str, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.3,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
