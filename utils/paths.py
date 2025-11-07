from pathlib import Path

_MARKERS = ("pyproject.toml", ".git", "README.md")

def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in _MARKERS):
            return parent
    return p.parent  # fallback

REPO_ROOT = find_repo_root()

def norm(p: str | Path) -> str:
    """Devuelve una ruta ABSOLUTA anclada al root del repo."""
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p))
