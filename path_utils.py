from __future__ import annotations

from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent


def _unique(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def resolve_input_path(value: str | Path | None, default_name: str = "matrix_ml_dataset.csv") -> Path:
    """Resolve an input file independently of the process working directory.

    Search order:
    1. absolute/user-supplied path;
    2. path relative to the current working directory;
    3. path relative to this bundle directory;
    4. bundle/data/<filename>;
    5. bundle/<filename>.
    """
    raw = Path(value) if value else Path("data") / default_name
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            Path.cwd() / raw,
            BASE_DIR / raw,
            BASE_DIR / "data" / raw.name,
            BASE_DIR / raw.name,
        ])

    for candidate in _unique(candidates):
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {p.resolve(strict=False)}" for p in _unique(candidates))
    raise FileNotFoundError(
        "Dataset CSV was not found.\n"
        f"Requested value: {value!r}\n"
        "Searched locations:\n"
        f"{searched}\n\n"
        "For command-line use, pass --input /absolute/path/to/matrix_ml_dataset.csv.\n"
        "For Streamlit Cloud, run streamlit_app.py and upload the CSV in the interface, "
        "or commit data/matrix_ml_dataset.csv to the repository."
    )


def resolve_config_path(value: str | Path | None) -> Path:
    raw = Path(value) if value else Path("config.json")
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, BASE_DIR / raw, BASE_DIR / raw.name]
    for candidate in _unique(candidates):
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {p.resolve(strict=False)}" for p in _unique(candidates))
    raise FileNotFoundError(f"Configuration file was not found. Searched:\n{searched}")


def resolve_output_path(value: str | Path | None, default_relative: str) -> Path:
    raw = Path(value) if value else Path(default_relative)
    path = raw if raw.is_absolute() else BASE_DIR / raw
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
