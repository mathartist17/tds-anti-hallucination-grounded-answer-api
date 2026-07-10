## Notes

- Build command in render
`uv sync --frozen && uv cache prune --ci`
- Add the AIPIPE_TOKEN environment variable
- Start command in render
`uv run uvicorn main:app --host 0.0.0.0 --port $PORT --reload`
