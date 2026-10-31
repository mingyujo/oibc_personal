"""Monthly local cache; no prior experiment artifacts or absolute repo paths."""
from __future__ import annotations
import gc
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from . import RECIPE_VERSION
from . import features as f
from .io import RunLock, event, fingerprint, private_root, read, sha, write

SOURCE_FILES = ["sites.csv", "train/generation.parquet", "train/forecast_wa.parquet", "train/forecast_wb.parquet"]


def sources(root):
    root = Path(root).resolve()
    if not all((root / p).is_file() for p in SOURCE_FILES): raise ValueError("Point --data-dir to the participant dataset directory")
    return {p: sha(root / p) for p in SOURCE_FILES}


def feature_contract():
    return fingerprint({"version": RECIPE_VERSION, "names": f.names(), "implementation": sha(Path(f.__file__))})


def prepare(root, work, start=None, end=None, resume=False):
    root = Path(root).resolve(); work = private_root(work); cache = work / "cache"
    before = sources(root)
    if start is None or end is None:
        stamps = pd.read_parquet(root / "train/generation.parquet", columns=["ts"]).ts
        dates = (f.timestamps(stamps) - pd.Timedelta("1ns")).normalize()
        start = dates.min() if start is None else f.stamp(start)
        end = dates.max() if end is None else f.stamp(end)
    start, end = f.stamp(start).normalize(), f.stamp(end).normalize()
    contract = {"version": RECIPE_VERSION, "feature_contract": feature_contract(), "data_root": str(root),
                "source_hashes": before, "start": start.isoformat(), "end": end.isoformat(),
                "width": 262, "target": "energy/capacity_kw; absent or negative labels remain masked",
                "cache_rows": "only plant-days with at least one valid label; full24h forecast inputs retained"}
    cache.mkdir(parents=True, exist_ok=True)
    with RunLock(cache, resume):
        manifest_path = cache / "manifest.json"
        if manifest_path.exists():
            manifest = read(manifest_path)
            if manifest["contract"] != contract: raise ValueError("Cache contract changed; choose a new --work-dir")
            if not resume: raise FileExistsError("Cache exists; pass --resume to verify and reuse")
        else:
            manifest = {"contract": contract, "chunks": [], "complete": False}
            write(manifest_path, manifest)
        sites = f.load_sites(root)
        if not (cache / "sites.parquet").exists(): sites.to_parquet(cache / "sites.parquet", index=False)
        for month in pd.period_range(start.tz_localize(None), end.tz_localize(None), freq="M"):
            name = str(month); folder = cache / name
            previous = next((c for c in manifest["chunks"] if c["month"] == name), None)
            if previous:
                if not all(sha(folder / p) == digest for p, digest in previous["hashes"].items()): raise ValueError("Cached chunk integrity check failed")
                continue
            a = max(start, f.stamp(month.start_time).normalize()); b = min(end, f.stamp(month.end_time).normalize())
            estimate = len(pd.date_range(a, b)) * len(sites) * 24 * 262 * 4
            if shutil.disk_usage(work).free < estimate * 1.25 + 64 * 1024**2: raise OSError("Insufficient disk for the next cache chunk")
            x, y, days, audit = f.build(root, a, b, sites=sites)
            keep = np.isfinite(y).any(1); x, y, days = x[keep], y[keep], days.loc[keep].reset_index(drop=True)
            folder.mkdir(exist_ok=True)
            for file, values in (("X.npy", x), ("y.npy", y)):
                temp = folder / (file + ".tmp")
                with temp.open("wb") as stream: np.save(stream, values)
                temp.replace(folder / file)
            temp = folder / "days.tmp.parquet"; days.to_parquet(temp, index=False); temp.replace(folder / "days.parquet")
            item = {"month": name, "plant_days": len(days), "labels": int(np.isfinite(y).sum()), "audit": audit,
                    "hashes": {p: sha(folder / p) for p in ("X.npy", "y.npy", "days.parquet")}}
            manifest["chunks"].append(item); write(manifest_path, manifest)
            event("cache_month_complete", month=name, plant_days=len(days), labels=item["labels"])
            del x, y, days; gc.collect()
        if sources(root) != before: raise RuntimeError("Original data changed while preparing the cache")
        manifest["complete"] = True; manifest["sites_sha256"] = sha(cache / "sites.parquet"); write(manifest_path, manifest)
    return cache


def cache_info(work):
    cache = private_root(work) / "cache"; manifest = read(cache / "manifest.json")
    if not manifest["complete"] or manifest["contract"]["feature_contract"] != feature_contract(): raise ValueError("Cache is incomplete or feature implementation changed")
    if sha(cache / "sites.parquet") != manifest["sites_sha256"]: raise ValueError("Cached metadata changed")
    return cache, manifest


def chunks(work, start=None, end=None, verify=True):
    cache, manifest = cache_info(work)
    for info in manifest["chunks"]:
        folder = cache / info["month"]
        if verify and not all(sha(folder / name) == digest for name, digest in info["hashes"].items()): raise ValueError("Cached data changed")
        days = pd.read_parquet(folder / "days.parquet")
        take = np.ones(len(days), dtype=bool)
        if start is not None: take &= (days.day >= f.stamp(start)).to_numpy()
        if end is not None: take &= (days.day <= f.stamp(end)).to_numpy()
        if not take.any(): continue
        x = np.load(folder / "X.npy", mmap_mode="r"); y = np.load(folder / "y.npy", mmap_mode="r")
        yield np.asarray(x[take]), np.asarray(y[take]), days.loc[take].reset_index(drop=True)


def assemble(work, run, cutoff, recipe, history_days=None):
    """A private memmap avoids a permanent second copy of the training history."""
    cache, manifest = cache_info(work); cutoff = f.stamp(cutoff).normalize()
    start = cutoff - pd.Timedelta(days=history_days) if history_days else None
    end = cutoff - pd.Timedelta("1d"); selected = f.recipe_indices(recipe)
    meta = pd.read_parquet(cache / "sites.parquet").set_index("site_id")
    frames = []
    for item in manifest["chunks"]:
        days = pd.read_parquet(cache / item["month"] / "days.parquet")
        take = (days.day <= end)
        if start is not None: take &= days.day >= start
        frames.append(days.loc[take])
    days = pd.concat(frames, ignore_index=True)
    if days.empty: raise ValueError("No observed training days before cutoff")
    estimate = len(days) * 24 * (len(selected) * 4 + 8)
    if shutil.disk_usage(run).free < estimate * 1.1 + 64 * 1024**2: raise OSError("Insufficient disk for training staging; use a larger work drive")
    x = np.lib.format.open_memmap(run / "working-X.npy", mode="w+", dtype="float32", shape=(len(days), 24, len(selected)))
    y = np.lib.format.open_memmap(run / "working-y.npy", mode="w+", dtype="float64", shape=(len(days), 24))
    offset = 0
    for xx, yy, dd in chunks(work, start, end):
        n = len(dd); x[offset:offset+n] = xx[:, :, selected]; y[offset:offset+n] = yy; offset += n
    if offset != len(days): raise ValueError("Training cache alignment failed")
    x.flush(); y.flush()
    cap = days.site_id.map(meta.capacity_kw).to_numpy(dtype="float64")
    return x, y, days, cap
