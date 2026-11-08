# ghost

Remembers what happens in a room from a camera and answers questions about it, e.g. "where did I leave my keys?".

## Run

```sh
cp .env.example .env                    # add OPENROUTER_API_KEY; optionally TZ=Area/City
cp cameras.example.yaml cameras.yaml    # set the camera's host and password
docker compose up -d --build
```

Open http://localhost:8765.

On first start the embeddings service downloads Qwen3-VL-Embedding-2B (about 4 GB).

## Test

```sh
.venv/bin/python -m pytest              # add -m "slow or not slow" for the long tracker scenarios
```

The tests need Postgres with pgvector on the PATH. They use no camera or model access: `tests/test_end_to_end.py` feeds synthetic frames through ingest, every background job and the agent's tools.
