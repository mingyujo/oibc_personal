"""All row-level CSV exports remain under .local-only/details/."""
from pathlib import Path


def write_csv(frame, path, *, root, **kwargs):
    root = Path(root).resolve()
    if root.name != ".local-only": raise ValueError("CSV root must be .local-only")
    target = root / "details" / Path(path)
    target = target.resolve()
    target.relative_to(root / "details")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, **kwargs)
    return target
