# The model is mocked, so these run without torch or sentence-transformers.
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts])


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(main, "_model", FakeModel())


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health_and_root(client):
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_embed_returns_one_vector_per_text(client, monkeypatch):
    monkeypatch.delenv("ML_API_KEY", raising=False)
    r = client.post("/embed", json={"texts": ["in the beginning", "amen"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dimensions"] == 3
    assert [e[0] for e in body["embeddings"]] == [16.0, 4.0]


def test_texts_are_stripped(client, monkeypatch):
    monkeypatch.delenv("ML_API_KEY", raising=False)
    r = client.post("/embed", json={"texts": ["  amen  "]})
    assert r.json()["embeddings"][0][0] == 4.0


@pytest.mark.parametrize("payload", [
    {},
    {"texts": []},
    {"texts": [""]},
    {"texts": ["   "]},
    {"texts": ["x" * 2001]},
    {"texts": ["ok"] * 257},
    {"texts": "not a list"},
    {"texts": [123]},
])
def test_bad_payloads_are_422(client, monkeypatch, payload):
    monkeypatch.delenv("ML_API_KEY", raising=False)
    assert client.post("/embed", json=payload).status_code == 422


def test_limits_are_inclusive(client, monkeypatch):
    monkeypatch.delenv("ML_API_KEY", raising=False)
    assert client.post("/embed", json={"texts": ["x" * 2000] * 256}).status_code == 200


def test_oversized_body_is_413(client, monkeypatch):
    monkeypatch.delenv("ML_API_KEY", raising=False)
    big = {"texts": ["x" * 2000] * 600}
    assert client.post("/embed", json=big).status_code == 413


def test_bogus_content_length_is_413(client):
    r = client.post("/embed", content=b"{}",
                    headers={"content-length": "abc", "content-type": "application/json"})
    assert r.status_code == 413


def test_api_key_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("ML_API_KEY", "s3cret")
    body = {"texts": ["amen"]}
    assert client.post("/embed", json=body).status_code == 401
    assert client.post("/embed", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/embed", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_non_ascii_key_is_401_not_500(client, monkeypatch):
    monkeypatch.setenv("ML_API_KEY", "s3cret")
    r = client.post("/embed", json={"texts": ["amen"]},
                    headers={"X-API-Key": "sécret".encode("utf-8")})
    assert r.status_code == 401


def test_health_needs_no_key(client, monkeypatch):
    monkeypatch.setenv("ML_API_KEY", "s3cret")
    assert client.get("/health").status_code == 200


def test_model_errors_do_not_leak_details(client, monkeypatch):
    monkeypatch.delenv("ML_API_KEY", raising=False)

    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("secret path /srv/model/weights.bin")

    monkeypatch.setattr(main, "_model", Broken())
    r = client.post("/embed", json={"texts": ["amen"]})
    assert r.status_code == 500
    assert r.json() == {"detail": "embedding failed"}


def test_cors_allows_only_the_site(client):
    ok = client.options("/embed", headers={
        "Origin": "https://scripturestack.j4den.com",
        "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://scripturestack.j4den.com"
    bad = client.options("/embed", headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
