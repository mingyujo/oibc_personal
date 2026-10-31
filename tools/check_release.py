"""Positive allowlist for publishing code, configs, docs and synthetic tests."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
TOP = {".gitignore", ".gitattributes", "AGENTS.md", "README.md", "pyproject.toml", "requirements-tested.txt", "setup.ps1", "setup.sh"}
HISTORY_FILES = {"learning_history.md", "learning_history_02.md"}
IGNORED = {".git", ".local-only", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}


def allowed(relative):
    if relative.as_posix() in TOP | HISTORY_FILES: return True
    parts = relative.parts
    if len(parts) != 2: return relative.as_posix() == ".github/workflows/tests.yml"
    if parts[0] in {"oibc_portable", "tests", "tools"}: return relative.suffix == ".py"
    if parts[0] == "configs": return relative.name in {"quick_cpu.json", "confirmation_cpu.json", "testbed_last_year.json"}
    if parts[0] == "docs": return relative.name in {"MODEL_REVIEW.md", "VALIDATION.md", "HISTORY.md", "HISTORY_MANIFEST.json", "TESTBED.md", "COMPETITION_CONTRACT.json"}
    return False


def audit(root=ROOT):
    root = Path(root).resolve(); files = {}; problems = []
    secret = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    real_key = re.compile(r"\b(?:PV|WA|WB)\d{3}\b")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts): continue
        if path.is_symlink(): problems.append({"path": relative.as_posix(), "reason": "symlink"}); continue
        if not path.is_file(): continue
        if not allowed(relative): problems.append({"path": relative.as_posix(), "reason": "not_allowlisted"}); continue
        content = path.read_bytes()
        limit = 1024 * 1024 if relative.as_posix() in HISTORY_FILES else 512 * 1024
        if len(content) > limit: problems.append({"path": relative.as_posix(), "reason": "too_large"}); continue
        try: text = content.decode("utf-8-sig")
        except UnicodeDecodeError: problems.append({"path": relative.as_posix(), "reason": "not_utf8"}); continue
        if secret.search(text) or real_key.search(text) or str(Path.home()) in text:
            problems.append({"path": relative.as_posix(), "reason": "sensitive_content_pattern"}); continue
        files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return {"passed": not problems, "files": files, "problems": problems, "source_data_included": False, "model_weights_included": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path); args = parser.parse_args()
    result = audit()
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["passed"] else 1)
