"""
Scripture Stack ML service.

FastAPI service exposing sentence embeddings over scripture text. Runs on
Railway, called from the Next.js app at scripturestack.j4den.com.
"""
from __future__ import annotations

import logging
import os
from typing import Annotated, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

logger = logging.getLogger("scripturestack-ml")

app = FastAPI(
    title="Scripture Stack ML",
    description="Sentence embeddings for biblical text.",
    version="0.1.0",
)

# CORS: allow the production frontend and localhost dev.
_allowed_origins = [
    "https://scripturestack.j4den.com",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Lazy-loaded model — keeps cold start fast, model loads on first request.
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
        _model = SentenceTransformer(model_name)
    return _model


# Verses and search queries are short. 2000 chars is roomy for either;
# anything bigger is a caller bug, not a real request.
EmbedText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class EmbedRequest(BaseModel):
    texts: List[EmbedText] = Field(..., min_length=1, max_length=256)


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: List[List[float]]


@app.get("/")
def root():
    return {"service": "scripturestack-ml", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    try:
        model = get_model()
        vectors = model.encode(req.texts, normalize_embeddings=True).tolist()
        return EmbedResponse(
            model=os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            dimensions=len(vectors[0]) if vectors else 0,
            embeddings=vectors,
        )
    except Exception as exc:
        # Full detail goes to the server log; the client just gets a 500.
        logger.exception("embedding failed")
        raise HTTPException(status_code=500, detail="embedding failed") from exc
