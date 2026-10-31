"""OIBC settlement and local submission checks. Inputs never leave this process."""
from pathlib import Path
import numpy as np
import pandas as pd
from . import features as f
from .io import sha

RULE_VERSION = "oibc-2026-local-contract-v1"
TEST_START, TEST_END, CUTOFF = "2024-06-01", "2025-05-31", "2024-05-31"


def ticks(values):
    try:
        items = [pd.Timestamp(value) for value in values]
        if any(pd.isna(value) or value.tz is None for value in items):
            raise ValueError("Explicit timezone and finite timestamps are required")
        result = pd.DatetimeIndex(pd.to_datetime(items, utc=True))
    except (ValueError, TypeError) as exc:
        raise ValueError("Timestamp parsing failed") from exc
    if result.tz is None or result.isna().any():
        raise ValueError("Explicit timezone and finite timestamps are required")
    result = result.tz_convert(f.TZ)
    if np.any(result.minute | result.second | result.microsecond | result.nanosecond):
        raise ValueError("Timestamps must be exact end-of-hour values")
    return result


def weights(times):
    return np.where(times.hour.isin([11, 12, 13, 14, 15]), 3., 1.)


def validate_sites(sites, profile="competition"):
    if profile not in {"competition", "synthetic"}: raise ValueError("Unknown testbed profile")
    if not {"site_id", "capacity_kw"} <= set(sites): raise ValueError("Site schema mismatch")
    if sites.site_id.isna().any() or sites.site_id.duplicated().any() or sites.empty:
        raise ValueError("Site keys must be complete and unique")
    cap = pd.to_numeric(sites.capacity_kw, errors="raise").to_numpy(float)
    if not np.isfinite(cap).all() or (cap <= 0).any(): raise ValueError("Invalid site capacity")
    if profile == "competition" and len(sites) != 253: raise ValueError("Competition profile requires253 sites")
    if profile == "synthetic" and not sites.site_id.astype(str).str.startswith("SYN-").all():
        raise ValueError("Synthetic profile accepts only invented fixture keys")


def composition(sites, incoming=None):
    capacity = pd.to_numeric(sites.capacity_kw, errors="raise").to_numpy(float)
    if sites.site_id.isna().any() or sites.site_id.duplicated().any() or not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("Invalid composition metadata")
    if incoming is None:
        if "current_group" not in sites or not sites.current_group.isin(["A", "B"]).all():
            raise ValueError("Reference composition requires A/B metadata")
        incoming = pd.DataFrame({"site_id": sites.site_id, "group": sites.current_group.map({"A": "G1", "B": "G2"})})
    if set(incoming) != {"site_id", "group"} or incoming.isna().any().any():
        raise ValueError("Composition requires exactly site_id and group")
    if incoming.site_id.duplicated().any() or set(incoming.site_id) != set(sites.site_id):
        raise ValueError("Composition must assign every site exactly once")
    if not incoming.group.isin(["G1", "G2"]).all(): raise ValueError("Unknown composition group")
    aligned = sites.site_id.map(incoming.set_index("site_id").group).to_numpy()
    capacities = {g: float(capacity[aligned == g].sum()) for g in ("G1", "G2")}
    if min(capacities.values()) < 20000: raise ValueError("Both groups require at least20000kW")
    return aligned, capacities


def price_series(path):
    frame = pd.read_csv(path)
    if not {"ts", "smp"} <= set(frame): raise ValueError("Price schema mismatch")
    times = ticks(frame.ts)
    prices = pd.to_numeric(frame.smp, errors="raise").to_numpy(float)
    if times.duplicated().any() or not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Prices must be unique, finite and nonnegative")
    if "k" in frame and not np.array_equal(pd.to_numeric(frame.k).to_numpy(), weights(times)):
        raise ValueError("Provided k differs from the end-hour11..15 rule")
    return pd.Series(prices, index=times)


def settlement(pred_kwh, truth_kwh, times, group, price):
    """Return aggregates and a common-support mask, never individual predictions.

    Arrays are time x site. Missing truth excludes the whole253-site hour;
    invalid predictions are errors, never a way to remove difficult hours.
    """
    pred, truth = np.asarray(pred_kwh, float), np.asarray(truth_kwh, float)
    times = ticks(times); group = np.asarray(group); price = np.asarray(price, float)
    if times.duplicated().any(): raise ValueError("Duplicate settlement hour")
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape != (len(times), len(group)):
        raise ValueError("Settlement array alignment failed")
    if not np.isfinite(pred).all() or (pred < 0).any(): raise ValueError("Predictions must be finite and nonnegative")
    if np.isinf(truth).any() or np.any(truth[np.isfinite(truth)] < 0): raise ValueError("Invalid truth values")
    if price.shape != (len(times),) or not np.isfinite(price).all() or (price < 0).any():
        raise ValueError("Missing or invalid evaluation prices")
    if set(group) != {"G1", "G2"}: raise ValueError("Settlement requires two fixed groups")
    valid = np.isfinite(truth).all(axis=1)
    revenue, penalty = np.zeros(len(times)), np.zeros(len(times))
    if valid.any():
        error = pred[valid] - truth[valid]
        group_abs = sum(abs(error[:, group == g].sum(axis=1)) for g in ("G1", "G2"))
        revenue[valid] = price[valid] * truth[valid].sum(axis=1)
        penalty[valid] = .1 * price[valid] * weights(times)[valid] * group_abs
    if not np.isfinite(revenue).all() or not np.isfinite(penalty).all(): raise ValueError("Settlement arithmetic overflow")
    count = int(valid.sum())
    return {"common_hours": count, "R": float(revenue.sum()) if count else None,
            "W": float(penalty.sum()) if count else None,
            "S": float((revenue-penalty).sum()) if count else None}, valid, revenue, penalty


def validate_submission(root, composition_file, forecast_file, profile="competition"):
    root = Path(root)
    sources = [root/"sites.csv", root/"smp.csv", Path(composition_file), Path(forecast_file)]
    before = [sha(p) for p in sources]
    sites = pd.read_csv(root/"sites.csv"); validate_sites(sites, profile)
    _, capacities = composition(sites, pd.read_csv(composition_file))
    official = price_series(root/"smp.csv")
    expected = pd.date_range("2025-06-01 01:00", periods=8760, freq="h", tz=f.TZ)
    if len(official) != 8760 or set(official.index) != set(expected):
        raise ValueError("Provided official grid must match the8760 prescribed hours")
    forecast = pd.read_csv(forecast_file)
    if set(forecast) != {"group", "ts", "forecast_kwh"} or len(forecast) != 17520:
        raise ValueError("Forecast requires exact columns and17520 rows")
    if forecast.isna().any().any() or not forecast.group.isin(["G1", "G2"]).all():
        raise ValueError("Forecast contains missing fields or unknown groups")
    forecast["ts"] = ticks(forecast.ts)
    values = pd.to_numeric(forecast.forecast_kwh, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or forecast.duplicated(["group", "ts"]).any():
        raise ValueError("Forecast values/keys are invalid")
    if not all(len(rows) == 8760 and set(rows.ts) == set(expected) for _, rows in forecast.groupby("group")):
        raise ValueError("Each group must cover the exact official grid")
    if [sha(p) for p in sources] != before: raise RuntimeError("Submission inputs changed during checking")
    return {"valid": True, "rule_version": RULE_VERSION, "profile": profile, "sites": len(sites),
            "group_capacity_kw": capacities, "hours_per_group": 8760, "forecast_rows": 17520,
            "file_hashes": dict(zip(("sites", "official_grid", "composition", "forecast"), before)),
            "official_score_available": False, "external_submission": False}
