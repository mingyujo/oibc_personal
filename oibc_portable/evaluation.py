"""Development metrics and local submission exports; no automatic uploading."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from . import data, features as f, models
from .io import event, private_root, read, sha, write
from .local_details import write_csv


class Totals:
    def __init__(self): self.absolute = 0.; self.bias = 0.; self.actual = 0.; self.exposure = 0.; self.labels = 0
    def add(self, pred, y, cap):
        good = np.isfinite(y); weight = np.broadcast_to(cap[:, None], y.shape)
        self.absolute += float((np.where(good, abs(pred-y), 0) * weight).sum())
        self.bias += float((np.where(good, pred-y, 0) * weight).sum())
        self.actual += float((np.where(good, y, 0) * weight).sum())
        self.exposure += float(weight[good].sum()); self.labels += int(good.sum())
    def result(self):
        return {"labels": self.labels, "E": self.absolute/self.exposure if self.exposure else None,
                "MAE_kwh": self.absolute/self.labels if self.labels else None,
                "WAPE_pct": 100*self.absolute/self.actual if self.actual > 0 else None,
                "bias_normalized": self.bias/self.exposure if self.exposure else None}


def settlement(pred, y, days, sites, prices):
    dates = pd.DatetimeIndex(sorted(days.day.unique())); di = dates.get_indexer(days.day)
    si = pd.Index(sites.site_id).get_indexer(days.site_id)
    if (si < 0).any(): raise ValueError("Unmapped settlement site")
    def matrix(a):
        out = np.full((len(dates)*24, len(sites)), np.nan)
        out[(di[:, None]*24 + np.arange(24)).reshape(-1), np.repeat(si, 24)] = a.reshape(-1)
        return out
    pp, yy = matrix(pred), matrix(y); complete = np.isfinite(yy).all(1) & np.isfinite(pp).all(1)
    if not complete.any(): return {"W": None, "common_hours": 0}
    ticks = pd.to_datetime(np.repeat(dates.as_unit("ns").asi8, 24) + np.tile(np.arange(1, 25), len(dates))*pd.Timedelta("1h").value, utc=True).tz_convert(f.TZ)
    price = prices.reindex(ticks).to_numpy()
    if not np.isfinite(price[complete]).all() or (price[complete] < 0).any(): raise ValueError("Invalid settlement price coverage")
    errors = (pp[complete]-yy[complete]) * sites.capacity_kw.to_numpy()[None, :]
    first = sites.current_group.eq("A").to_numpy()
    k = np.where(ticks.hour.isin([11,12,13,14,15]), 3., 1.)
    W = .1 * (price[complete]*k[complete]*(abs(errors[:, first].sum(1)) + abs(errors[:, ~first].sum(1)))).sum()
    return {"W": float(W), "common_hours": int(complete.sum())}


def evaluate(work, model_dir, start, end, device="cpu"):
    work = private_root(work); cache, manifest = data.cache_info(work)
    net, config, contract = models.load(model_dir, device)
    if contract["source_hashes"] != manifest["contract"]["source_hashes"]: raise ValueError("Evaluation dataset differs from the training data contract")
    start, end = f.stamp(start).normalize(), f.stamp(end).normalize()
    if start <= f.stamp(config.cutoff): raise ValueError("Evaluation must start after the frozen model cutoff")
    if start < f.stamp(manifest["contract"]["start"]) or end > f.stamp(manifest["contract"]["end"]): raise ValueError("Requested evaluation is outside the prepared cache")
    root = Path(manifest["contract"]["data_root"]); sites = pd.read_parquet(cache / "sites.parquet"); meta = sites.set_index("site_id")
    prices = pd.read_csv(root / "train/smp.csv"); prices["ts"] = f.timestamps(prices.ts)
    if prices.ts.duplicated().any(): raise ValueError("Duplicate price timestamp")
    prices = prices.set_index("ts").smp
    all_rows, day_rows, night_rows = Totals(), Totals(), Totals(); monthly = []; W = 0.; common = 0
    for x, y, days in data.chunks(work, start, end):
        p = models.predict(net, config, x, model_dir, device); cap = days.site_id.map(meta.capacity_kw).to_numpy()
        all_rows.add(p, y, cap); daylight = x[:, :, 212] > 0
        day_rows.add(p, np.where(daylight, y, np.nan), cap); night_rows.add(p, np.where(~daylight, y, np.nan), cap)
        part = Totals(); part.add(p, y, cap); money = settlement(p, y, days, sites, prices)
        if money["W"] is not None: W += money["W"]; common += money["common_hours"]
        monthly.append({"month": days.day.iloc[0].strftime("%Y-%m"), **part.result(), **money})
    result = {"development_evaluation": True, "independent_final_test": False, "model": config.model, "recipe": config.recipe,
              "cutoff": config.cutoff, "start": start.isoformat(), "end": end.isoformat(), "all": all_rows.result(),
              "day": day_rows.result(), "night": night_rows.result(), "settlement": {"W": W if common else None, "common_hours": common},
              "monthly": monthly, "no_missing_targets_filled": True}
    if not result["all"]["labels"]: raise ValueError("No evaluation labels in the requested interval")
    output = work / "evaluations" / Path(model_dir).name / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + ".json")
    write(output, result); event("evaluation_complete", model=config.model, recipe=config.recipe, **result["all"], common_settlement_hours=common)
    return output


def predict_submission(root, work, model_dir, device="cpu", composition=None):
    root = Path(root).resolve(); work = private_root(work); net, config, contract = models.load(model_dir, device)
    if sha(root / "sites.csv") != contract["source_hashes"]["sites.csv"]: raise ValueError("Prediction metadata differs from training")
    original_hashes = {name: sha(root / name) for name in ("sites.csv", "smp.csv", "test/forecast_wa.parquet", "test/forecast_wb.parquet")}
    sites = f.load_sites(root)
    if composition is not None:
        incoming = pd.read_csv(composition)
        if not {"site_id", "group"} <= set(incoming) or incoming.site_id.duplicated().any() or set(incoming.site_id) != set(sites.site_id) or not incoming.group.isin(["G1", "G2"]).all(): raise ValueError("Invalid composition file")
        sites["current_group"] = sites.site_id.map(incoming.set_index("site_id").group).map({"G1":"A", "G2":"B"})
    capacities = sites.groupby("current_group").capacity_kw.sum()
    if set(capacities.index) != {"A", "B"} or (capacities < 20000).any(): raise ValueError("Each group must contain at least20000kW")
    raw_times = pd.read_csv(root / "smp.csv", usecols=["ts"])
    ticks = f.timestamps(raw_times.ts)
    if len(ticks) != 8760 or ticks.duplicated().any(): raise ValueError("Expected8760 unique official timestamps")
    target_days = (ticks - pd.Timedelta("1ns")).normalize()
    first_origin = target_days.min() - pd.Timedelta("1d") + pd.Timedelta("17h")
    if f.stamp(config.cutoff) > first_origin: raise ValueError("Model trained after first official forecast origin")
    meta = sites.set_index("site_id"); parts = []
    for month in sorted(set(target_days.strftime("%Y-%m"))):
        keep = target_days.strftime("%Y-%m") == month; dates = target_days[keep]
        x, _, days, _ = f.build(root, dates.min(), dates.max(), split="test", labels=False, sites=sites)
        pred = models.predict(net, config, x, model_dir, device)
        energy = pred * days.site_id.map(meta.capacity_kw).to_numpy()[:, None]
        rows = pd.DataFrame({"ts": pd.to_datetime(np.repeat(pd.DatetimeIndex(days.day).as_unit("ns").asi8, 24) + np.tile(np.arange(1,25), len(days))*pd.Timedelta("1h").value, utc=True).tz_convert(f.TZ),
                             "group": np.repeat(days.site_id.map(meta.current_group).map({"A":"G1", "B":"G2"}).to_numpy(), 24),
                             "forecast_kwh": energy.reshape(-1)})
        parts.append(rows.groupby(["group", "ts"], as_index=False).forecast_kwh.sum())
        event("official_month_predicted", month=month)
    forecasts = pd.concat(parts, ignore_index=True).sort_values(["group", "ts"]).reset_index(drop=True)
    if len(forecasts) != 17520 or forecasts.duplicated(["group", "ts"]).any() or not np.isfinite(forecasts.forecast_kwh).all() or (forecasts.forecast_kwh < 0).any(): raise ValueError("Invalid submission forecast grid")
    if not all(set(g.ts) == set(ticks) for _, g in forecasts.groupby("group")): raise ValueError("Submission timestamps differ from provided SMP grid")
    groups = pd.DataFrame({"site_id": sites.site_id, "group": sites.current_group.map({"A":"G1", "B":"G2"})})
    tag = Path(model_dir).name + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    a = write_csv(groups, Path(tag) / "composition.csv", root=work, index=False)
    b = write_csv(forecasts, Path(tag) / "forecast.csv", root=work, index=False)
    info = {"external_submission": False, "sites": len(sites), "forecast_rows": len(forecasts), "hours_per_group": 8760,
            "files": {a.name: sha(a), b.name: sha(b)}, "model_sha256": read(Path(model_dir) / "complete.json")["model_sha256"]}
    if any(sha(root / name) != digest for name, digest in original_hashes.items()): raise RuntimeError("Original prediction inputs changed during inference")
    info["source_hashes"] = original_hashes
    write(a.parent / "verification.json", info); event("local_submission_files_ready", sites=len(sites), forecast_rows=len(forecasts), directory=str(a.parent))
    return a.parent
