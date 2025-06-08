# Coding Guidelines for pl-voiceagent

## Project Structure
- Root of the repo contains all application source files such as `main.py`, `dpg.py`, `ell.py`, `oai.py`, `twilio.py`, `utils.py`, and `vad_events.py`.
- `services/` holds helper modules. Currently it only contains `cache.py`.
- `.github/workflows/` contains the CI configuration (`deploy.yml`) used for Docker builds and deployment.
- `fastmcp.md` provides documentation.
- There is currently no dedicated `tests/` directory or automated test suite.

## Test Commands
- No automated tests are present. When tests are added, run them with `pytest`.
- To verify builds locally, you can build the Docker image:
  ```bash
  docker build -f dockerfile .
  ```

## Coding Conventions
- Follow [PEP 8](https://peps.python.org/pep-0008/) standards.
- Use 4 spaces for indentation.
- Function and variable names should use `snake_case`; classes should use `CamelCase`.
- Keep line length under 120 characters when possible.
- Include docstrings for public modules, classes, and functions.

## Pull Request Guidelines
- Use descriptive titles prefixed with tags such as `[Fix]`, `[Feature]`, `[Docs]`, or `[Refactor]`.
- The PR body must contain the following sections:
  - `Summary` – short description of the change.
  - `Testing Done` – commands or steps taken to validate the change.
- Ensure your branch is up to date with `main` before opening a PR.

## Programmatic Checks
- CI: `.github/workflows/deploy.yml` builds and pushes a Docker image on pushes to `main`.
- Before committing, ensure Python files compile:
  ```bash
  python -m py_compile *.py
  ```
- Run any available linters (e.g., `flake8`) and formatters (e.g., `black`). While not enforced by CI, they help maintain code quality.

