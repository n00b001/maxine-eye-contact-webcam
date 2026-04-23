# Project Rules

## Development Workflow

- Branch from `main` for all changes.
- Open a pull request before merging.
- All CI checks must pass before merge.

## Code Style

- Format and lint with **ruff**.
- Maximum line length: **100**.

## Testing Requirements

- Use **pytest** for all tests.
- Minimum code coverage: **80%**.

## Commit Hooks

- **pre-commit** runs `ruff check`, `ruff format`, and coverage checks.
- **pre-push** runs the full test suite.

## CI/CD Requirements

- All GitHub Actions checks must pass before merging.

## Adding Dependencies

- Use `uv add <package>` to add runtime dependencies.
- Use `uv add --group dev <package>` for dev-only tools.
- Use `uv add --optional <extra-name> <package>` for opt-in feature extras
  (installable via `uv sync --extra <extra-name>`).
- **Do NOT use `uv pip install`** — it bypasses `pyproject.toml` and
  `uv.lock`, so deps end up in the venv only and silently disappear on the
  next `uv sync`. This has been violated twice; call it out if you see it.
- **Do NOT use `pip` / `pip install` directly** — same reason.
- The only exception is for packages whose pins are incompatible with the
  project's locked deps (e.g. `mediapipe` pinning `protobuf < 5`). In that
  case, keep the package entirely out of `pyproject.toml` and document the
  manual `uv pip install` command in the README for end users only — but
  our own tooling must still drive through `uv add` for everything we own.
