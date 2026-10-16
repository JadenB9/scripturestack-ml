# scripturestack-ml

The Python half of [Scripture Stack](https://scripturestack.j4den.com).

The site's semantic search works by comparing meaning rather than keywords,
and that means turning text into vectors. This is the service that does it:
a small FastAPI app wrapping sentence-transformers. The Next.js frontend
sends it verses (or your search query), it sends back embeddings, and the
site compares them with pgvector to find verses that mean similar things.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check (used by Railway) |
| POST | `/embed` | Return sentence embeddings for a list of texts |

`/embed` takes `{"texts": ["...", "..."]}`: 1 to 256 strings, each 1 to 2000
characters after trimming, and a request body of at most 1 MB (413 past that).
It returns `{"model", "dimensions", "embeddings"}` with L2-normalized vectors,
one per text. Bad input is a 422. An internal failure is a plain 500 with no
details; the real error only goes to the server log.

## Auth

If `ML_API_KEY` is set, `/embed` requires it in the `X-API-Key` header and
answers 401 otherwise. If it is not set, `/embed` is open, which is meant for
local development only. Set it in the deploy and in the Vercel project. `/` and
`/health` never need a key. CORS only allows `scripturestack.j4den.com` and
`localhost:3000`, though the Next.js app calls this server-side anyway.

There is no rate limiting in the service itself. The key is what keeps strangers
from driving inference.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Service runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.
The first `/embed` call downloads and loads the model (`EMBED_MODEL`, default
`all-mpnet-base-v2`, 768 dimensions), so it is slow once.

## Tests

```bash
pip install fastapi==0.141.1 pydantic==2.13.4 -r requirements-dev.txt   # no torch needed
ruff check .
pytest
```

The tests replace the model with a stub, so they run without torch or a model
download. They cover validation limits, the API key check, the body size cap,
CORS, and that errors do not leak details. CI runs the same.

## Deployment

Deployed to Railway. The deployed URL is set as `ML_API_URL` in the Vercel project for `scripturestack`.

```bash
railway login
railway link    # link to the scripturestack-ml project
railway up      # deploy
```

## License

MIT, see `LICENSE`.
