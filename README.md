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

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Service runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Deployment

Deployed to Railway. The deployed URL is set as `ML_API_URL` in the Vercel project for `scripturestack`.

```bash
railway login
railway link    # link to the scripturestack-ml project
railway up      # deploy
```
