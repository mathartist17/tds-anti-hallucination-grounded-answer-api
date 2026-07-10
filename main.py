import os
import json
from typing import List

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

app = FastAPI()

# Allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "openai/gpt-4o"
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]

SYSTEM_PROMPT = """You are a strict grounded question-answering engine for a compliance system.

RULES:
1. Only use information found in the provided context chunks. Never use outside
   knowledge, assumptions, or general world knowledge.
2. Every factual claim in your answer must come from a chunk, and you must list
   the chunk_id(s) that support your answer in "citations".
3. If the answer is not clearly supported by the chunks, you MUST treat the
   question as unanswerable.
4. Output ONLY a single JSON object - no explanations, no markdown fences,
   no extra text before or after it.

Output JSON shape:
{"answer": "<string>", "citations": ["<chunk_id>", ...], "confidence": <float 0-1>, "answerable": <true|false>}

Confidence calibration:
- 0.90-1.00: answer is stated explicitly and unambiguously in the chunks
- 0.50-0.89: answer requires light inference or combining 2+ chunks
- 0.00-0.30: unanswerable -> answer must be exactly "I don't know", citations [],
  answerable false
"""


class Chunk(BaseModel):
    chunk_id: str
    text: str


class QueryRequest(BaseModel):
    question: str
    chunks: List[Chunk]


FALLBACK = {
    "answer": "I don't know",
    "citations": [],
    "confidence": 0.0,
    "answerable": False,
}


def build_user_prompt(question: str, chunks: List[Chunk]) -> str:
    chunk_block = "\n".join(
        f"[{chunk.chunk_id}]: {chunk.text}" for chunk in chunks
    )

    return (
        f"Context chunks:\n{chunk_block}\n\n"
        f"Question: {question}\n\n"
        f"Respond with ONLY the JSON object described in the system prompt."
    )


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/grounded-qa")
async def grounded_qa(request: Request):
    # Step 1: Validate input
    try:
        raw_body = await request.json()
        payload = QueryRequest(**raw_body)
    except (json.JSONDecodeError, ValidationError, Exception):
        return FALLBACK

    if not payload.question.strip() or not payload.chunks:
        return FALLBACK

    valid_ids = {chunk.chunk_id for chunk in payload.chunks}

    # Step 2: Query AI Pipe
    try:
        response = requests.post(
            "https://aipipe.org/openrouter/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {AIPIPE_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            payload.question,
                            payload.chunks,
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        raw_text = (
            data["choices"][0]["message"]["content"]
            .strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        result = json.loads(raw_text)

    except Exception:
        return FALLBACK

    # Step 3: Validate model output
    citations = [
        c for c in result.get("citations", [])
        if c in valid_ids
    ]

    answerable = (
        bool(result.get("answerable", False))
        and len(citations) > 0
    )

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    answer = result.get("answer") or "I don't know"

    if not answerable:
        answer = "I don't know"
        citations = []
        confidence = min(confidence, 0.3)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "answerable": answerable,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
