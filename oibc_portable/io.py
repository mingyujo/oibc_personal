"""Atomic local artifacts and process locks; console output is aggregate only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import math
import threading
from datetime import datetime, timezone
import psutil

_event_file = None
_event_lock = threading.Lock()


def clean(value):
    import numpy as np
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(12):
        try: os.replace(temp, path); return
        except PermissionError:
            if attempt == 11: raise
            time.sleep(.1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""): digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def private_root(path):
    path = Path(path).resolve()
    if path.name != ".local-only": path /= ".local-only"
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitignore").write_text("*\n", encoding="ascii")
    return path


def event(name, **aggregates):
    text = json.dumps(clean(dict(time_utc=datetime.now(timezone.utc).isoformat(), event=name, **aggregates)), ensure_ascii=True)
    with _event_lock:
        print(text, flush=True)
        if _event_file is not None:
            with _event_file.open("a", encoding="utf-8") as stream: stream.write(text + "\n")


def set_event_file(path):
    global _event_file
    _event_file = Path(path)


class Heartbeat:
    def __init__(self, model, stage): self.model=model; self.stage=stage; self.stop=threading.Event()
    def __enter__(self):
        self.start=time.monotonic()
        def beat():
            while not self.stop.wait(30):
                event("fit_alive",model=self.model,stage=self.stage,elapsed_seconds=time.monotonic()-self.start,
                      pid=os.getpid(),resident_RAM_GiB=psutil.Process().memory_info().rss/2**30)
        self.thread=threading.Thread(target=beat,daemon=True); self.thread.start(); return self
    def __exit__(self,*_): self.stop.set(); self.thread.join(timeout=1)


class RunLock:
    """Reject concurrent jobs; stale locks require explicit resume."""
    def __init__(self, folder, resume=False):
        self.path = Path(folder) / "run.lock"
        self.resume = resume

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            old = read(self.path)
            try:
                proc = psutil.Process(old["pid"])
                live = abs(proc.create_time() - old["created"]) < 1
            except psutil.NoSuchProcess: live = False
            if live: raise RuntimeError("Another process owns this run")
            if not self.resume: raise RuntimeError("Stale lock; use --resume after checking the prior job")
            self.path.rename(self.path.with_name(f"run-stale-{time.time_ns()}.lock"))
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created": psutil.Process().create_time(), "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
        return self

    def __exit__(self, *_): self.path.unlink(missing_ok=True)
