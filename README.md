# scripturestack-ml

Python ML service for [Scripture Stack](https://scripturestack.j4den.com). Provides sentence embeddings, sentiment, and stylometric features for biblical text. Called from the Next.js frontend.

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
