from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3-32b"


def get_model() -> ChatGroq:
    return ChatGroq(
        model=os.environ.get("GROQ_MODEL_NAME", DEFAULT_MODEL),
        temperature=0,
    )
