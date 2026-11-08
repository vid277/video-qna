# AGENTS.md

- Keep comments to a minimum; the code should explain itself.
- Add blank lines between logical steps inside functions.
- Use strict pydantic models (`Record`: strict, frozen, no extra fields) for all data.
- Avoid `if` statements; prefer argparse, enums, dict lookups, and set operations.
- Models are fixed, flat, vendor-neutral table interfaces keyed by `camera_id`; don't change them per vendor.
- All camera models live in `models/camera.py`.
- Camera settings come from `cameras.yaml`, not code or env vars.
- Secrets live in `.env` (never in YAML or code); YAML points to it with `env_file`.
- Each camera brand subclasses the `Camera` ABC in `camera.py` (e.g. `reolink.py`).
- Run with `.venv/bin/python ghostcam.py ...`; `pyright` must pass with 0 errors.
- Run `ghostcam.py ingest` (capture) and `ghostcam.py worker` (embeddings, captions, summaries, cleanup) as separate processes.
- Background work is a `JobKind` in `ingest/tasks.py`: its state is derived from Postgres, jobs must be idempotent, and the Huey queue (`jobs/`) only carries keys.
