"""Automatic .env loading and ClearML credential discovery.

The entry points call :func:`load_dotenv` once so that ClearML variables placed
in a repository ``.env`` file reach ``os.environ`` before ``Task.init`` runs.
:func:`clearml_is_configured` then decides whether ClearML logging is available
without any manual ``clearml-init`` step.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The minimum ClearML variables needed to authenticate from the environment.
CLEARML_REQUIRED = ("CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")


def load_dotenv(path: str | Path | None = None) -> None:
    """Populate ``os.environ`` from a KEY=VALUE ``.env`` file.

    Existing environment variables always win, so real shell/Compose values are
    never overwritten. A missing file is ignored.
    """

    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def clearml_is_configured() -> bool:
    """Return ``True`` when ClearML can authenticate automatically.

    True if the required ``CLEARML_*`` variables are present (typically loaded
    from ``.env``) or a ``~/clearml.conf`` from a manual ``clearml-init`` exists.
    """

    if all(os.environ.get(key) for key in CLEARML_REQUIRED):
        return True
    return (Path.home() / "clearml.conf").is_file()
