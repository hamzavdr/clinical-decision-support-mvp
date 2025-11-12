import os, json, requests
from typing import List, Dict

OPENAI_MODEL = None
OPENAI_URL = None

def configure(model: str, api_base: str):
    global OPENAI_MODEL, OPENAI_URL
    OPENAI_MODEL = model
    OPENAI_URL = f"{api_base}/chat/completions".rstrip("/")

def openai_chat(messages: List[Dict[str, str]], temperature=0.2) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
    r = requests.post(OPENAI_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
