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

- Use `uv add <package>` to add dependencies.
- Do **not** use `pip` directly.
