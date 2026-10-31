"""RAW6 feature contract. Future observations and prices are not arguments."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
WEATHER = {
    "wa": ["wa_cloud", "wa_temp", "wa_temp_max", "wa_temp_min", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_wind_dir", "wa_rain", "wa_snow"],
    "wb": ["wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point", "wb_wind_speed", "wb_wind_dir", "wb_wind_gust_spd", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_rain_prob", "wb_snow_prob", "wb_ice_prob", "wb_total_liq", "wb_rain", "wb_snow", "wb_ice", "wb_cloud"],
}
META = ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]
TIME = ["hour_sin", "hour_cos", "target_doy_sin", "target_doy_cos"]
VALUES = [f"{v}_r{r}" for p in WEATHER for r in (10, 17) for v in WEATHER[p]]
ROW_MISSING = [f"{p}_r{r}_missing" for p in WEATHER for r in (10, 17)]
QUALITY = [f"wa_r{r}_{q}" for r in (10, 17) for q in ("humidity_gt100", "temp_triplet_mismatch")]
BASE = META + TIME + VALUES + ROW_MISSING + [v + "_is_missing" for v in VALUES] + QUALITY
LOCAL_VARS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wb_cloud", "wb_temp", "wb_uv_idx", "wb_precip_prob"]
LOCAL = [BASE.index(f"{v}_r{r}") for v in LOCAL_VARS for r in (10, 17)]
OFFSETS = (-2, -1, 1, 2)
GEO = ["solar_sine_start", "solar_sine_mid", "solar_sine_end", "solar_positive_simpson", "solar_daylight_fraction_5points", "solar_geometric_daylength"]
COMMON = ["wa_temp", "wa_humidity", "wa_cloud", "wb_temp", "wb_rel_hum", "wb_cloud"]


def stamp(value):
    t = pd.Timestamp(value)
    return t.tz_localize(TZ) if t.tz is None else t.tz_convert(TZ)


def timestamps(values):
    t = pd.DatetimeIndex(pd.to_datetime(values))
    if t.isna().any(): raise ValueError("Missing timestamp")
    return t.tz_localize(TZ) if t.tz is None else t.tz_convert(TZ)


def names():
    out = list(BASE)
    out += [BASE[i] + f"|neighbor{off}" for off in OFFSETS for i in LOCAL]
    out += [f"neighbor_boundary{off}" for off in OFFSETS]
    out += ["wb_cloud_daily_variation", "wb_cloud_daily_mean", "wb_uv_idx_daily_mean"] + GEO
    out += [f"context|{v}|{kind}" for kind in ("current", "neighbor-2", "neighbor-1", "neighbor1", "neighbor2", "daily_mean", "daily_std", "finite_fraction") for v in COMMON]
    assert len(out) == 262 and len(set(out)) == 262
    return out


def recipe_indices(recipe):
    if recipe == "raw6": return np.arange(262)
    if recipe == "base137": return np.arange(137)
    if recipe == "no_weather": return np.array(list(range(9)) + list(range(208, 214)))
    raise ValueError("Unknown feature recipe")


def load_sites(root):
    sites = pd.read_csv(Path(root) / "sites.csv")
    required = ["site_id", "current_group", "wa_station", "wb_station"] + META
    if not set(required) <= set(sites): raise ValueError("Site metadata schema mismatch")
    if sites[required].isna().any().any() or sites.site_id.duplicated().any(): raise ValueError("Invalid site metadata keys")
    if not np.isfinite(sites[META].to_numpy()).all() or not sites.capacity_kw.gt(0).all(): raise ValueError("Invalid numeric metadata")
    if not sites.current_group.isin(["A", "B"]).all(): raise ValueError("Reference groups must be A/B")
    return sites.sort_values("site_id").reset_index(drop=True)


def forecast(root, provider, start, end, split="train"):
    path = Path(root) / split / f"forecast_{provider}.parquet"
    f = pd.read_parquet(path, filters=[("ts", ">=", start), ("ts", "<=", end)])
    station = provider + "_station"
    cols = [station, "round", "ts"] + WEATHER[provider]
    if not set(cols) <= set(f): raise ValueError("Forecast schema mismatch")
    f = f[cols].copy(); f["ts"] = timestamps(f.ts)
    if f[[station, "round", "ts"]].isna().any().any() or f.duplicated([station, "round", "ts"]).any(): raise ValueError("Duplicate or missing forecast key")
    if not f["round"].isin([10, 17]).all() or np.isinf(f[WEATHER[provider]].to_numpy()).any(): raise ValueError("Invalid forecast values")
    hourly = (f.ts.dt.minute == 0) & (f.ts.dt.second == 0) & (f.ts.dt.microsecond == 0) & (f.ts.dt.nanosecond == 0)
    return f.loc[hourly], int((~hourly).sum())


def solar_terms(times, lat, lon):
    t = timestamps(times); hour = t.hour.to_numpy() + t.minute.to_numpy() / 60 + t.second.to_numpy() / 3600
    gamma = 2 * np.pi / np.where(t.is_leap_year, 366., 365.) * (t.dayofyear.to_numpy() - 1 + (hour - 12) / 24)
    eq = 229.18 * (.000075 + .001868 * np.cos(gamma) - .032077 * np.sin(gamma) - .014615 * np.cos(2 * gamma) - .040849 * np.sin(2 * gamma))
    dec = .006918 - .399912 * np.cos(gamma) + .070257 * np.sin(gamma) - .006758 * np.cos(2 * gamma) + .000907 * np.sin(2 * gamma) - .002697 * np.cos(3 * gamma) + .00148 * np.sin(3 * gamma)
    la = np.deg2rad(lat); ha = np.deg2rad((hour * 60 + eq + 4 * np.asarray(lon) - 540) / 4 - 180)
    sine = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    length = 24 / np.pi * np.arccos(np.clip(-np.tan(la) * np.tan(dec), -1, 1))
    return np.clip(sine, -1, 1), length


def geometry(days, sites):
    meta = sites.set_index("site_id"); la = days.site_id.map(meta.lat).to_numpy(dtype="float64"); lo = days.site_id.map(meta.lon).to_numpy(dtype="float64")
    if not np.isfinite(la).all() or not np.isfinite(lo).all() or (abs(la) > 90).any() or (abs(lo) > 180).any(): raise ValueError("Invalid solar coordinates")
    result = np.empty((len(days), 24, 6), dtype="float32")
    for start in range(0, len(days), 2048):
        end = min(start + 2048, len(days)); dates = pd.DatetimeIndex(days.day.iloc[start:end]).as_unit("ns")
        ticks = np.repeat(dates.asi8, 24) + np.tile(np.arange(1, 25), len(dates)) * pd.Timedelta("1h").value
        t = pd.to_datetime(ticks, utc=True).tz_convert(TZ)
        parts = [solar_terms(t + pd.Timedelta(minutes=off), np.repeat(la[start:end], 24), np.repeat(lo[start:end], 24)) for off in (-60, -45, -30, -15, 0)]
        sine = np.stack([p[0] for p in parts], axis=-1)
        vals = np.column_stack([sine[:, 0], sine[:, 2], sine[:, 4], np.maximum(sine, 0) @ np.array([1, 4, 2, 4, 1]) / 12, (sine > 0).mean(1), parts[2][1]])
        result[start:end] = vals.reshape(-1, 24, 6)
    return result


def shift(x, off):
    out = np.full_like(x, np.nan)
    if off < 0: out[:, -off:] = x[:, :24 + off]
    else: out[:, :24 - off] = x[:, off:]
    return out


def expand(raw, geo):
    parts = [raw] + [shift(raw[:, :, LOCAL], off) for off in OFFSETS]
    boundary = np.stack([(np.arange(24) + off < 0) | (np.arange(24) + off > 23) for off in OFFSETS], axis=-1)
    parts.append(np.broadcast_to(boundary, (len(raw), 24, 4)).astype("float32"))
    cl = raw[:, 8:17, BASE.index("wb_cloud_r17")].astype("float64"); uv = raw[:, 8:17, BASE.index("wb_uv_idx_r17")].astype("float64")
    count = np.isfinite(cl).sum(1); mean = np.nansum(abs(cl), axis=1) / np.maximum(count, 1)
    delta = abs(np.diff(cl, axis=1)); variation = np.nansum(delta, axis=1) / np.maximum(np.isfinite(delta).sum(1), 1) / np.maximum(mean, 1.)
    uvmean = np.nansum(uv, axis=1) / np.maximum(np.isfinite(uv).sum(1), 1)
    variation[count < 2] = np.nan; mean[count == 0] = np.nan; uvmean[np.isfinite(uv).sum(1) == 0] = np.nan
    daily = np.column_stack([variation, mean, uvmean]).astype("float32")
    parts += [np.broadcast_to(daily[:, None, :], (len(raw), 24, 3)), geo]
    a = raw[:, :, [BASE.index(v + "_r17") for v in COMMON]]; b = raw[:, :, [BASE.index(v + "_r10") for v in COMMON]]
    anchor = np.where(np.isfinite(a), a, b)
    context = [anchor] + [shift(anchor, off) for off in OFFSETS]
    finite = np.isfinite(anchor); n = finite.sum(1, keepdims=True); safe = np.where(finite, anchor, 0).astype("float64")
    mu = np.divide(safe.sum(1, keepdims=True), n, out=np.full_like(safe[:, :1], np.nan), where=n > 0)
    var = np.divide(np.where(finite, (anchor - mu)**2, 0).sum(1, keepdims=True), n, out=np.full_like(mu, np.nan), where=n > 0)
    context += [np.broadcast_to(stat, anchor.shape).astype("float32") for stat in (mu, np.sqrt(var), n / 24)]
    parts += context
    out = np.concatenate(parts, axis=-1)
    if out.shape != (len(raw), 24, 262) or np.isinf(out).any(): raise ValueError("Invalid RAW6 feature matrix")
    return out


def build(root, start, end, *, split="train", labels=True, sites=None):
    """Build complete target-days; end is inclusive. Targets are separate from X."""
    start, end = stamp(start).normalize(), stamp(end).normalize()
    if end < start: raise ValueError("Invalid target-day range")
    sites = load_sites(root) if sites is None else sites.sort_values("site_id").reset_index(drop=True)
    days = pd.MultiIndex.from_product([pd.date_range(start, end, freq="D"), sites.site_id], names=["day", "site_id"]).to_frame(index=False)
    ticks = pd.to_datetime(np.repeat(pd.DatetimeIndex(days.day).as_unit("ns").asi8, 24) + np.tile(np.arange(1, 25), len(days)) * pd.Timedelta("1h").value, utc=True).tz_convert(TZ)
    meta = sites.set_index("site_id"); raw = np.empty((len(days), 24, 137), dtype="float32")
    for j, name in enumerate(META): raw[:, :, j] = days.site_id.map(meta[name]).to_numpy()[:, None]
    raw[:, :, 5] = np.sin(2 * np.pi * ticks.hour.to_numpy().reshape(-1, 24) / 24)
    raw[:, :, 6] = np.cos(2 * np.pi * ticks.hour.to_numpy().reshape(-1, 24) / 24)
    angle = 2 * np.pi * days.day.dt.dayofyear.to_numpy() / np.where(days.day.dt.is_leap_year, 366, 365)
    raw[:, :, 7] = np.sin(angle)[:, None]; raw[:, :, 8] = np.cos(angle)[:, None]
    excluded = {}
    for provider in WEATHER:
        f, excluded[provider] = forecast(root, provider, ticks.min(), ticks.max(), split)
        station = provider + "_station"
        keys = pd.MultiIndex.from_arrays([np.repeat(days.site_id.map(meta[station]).to_numpy(), 24), ticks], names=[station, "ts"])
        for rd in (10, 17):
            rows = f.loc[f["round"].eq(rd)].set_index([station, "ts"])
            values = rows[WEATHER[provider]].reindex(keys).to_numpy(dtype="float32").reshape(len(days), 24, -1)
            for j, variable in enumerate(WEATHER[provider]):
                name = f"{variable}_r{rd}"
                raw[:, :, BASE.index(name)] = values[:, :, j]
                raw[:, :, BASE.index(name + "_is_missing")] = ~np.isfinite(values[:, :, j])
            raw[:, :, BASE.index(f"{provider}_r{rd}_missing")] = (~keys.isin(rows.index)).reshape(-1, 24)
            if provider == "wa":
                hum = raw[:, :, BASE.index(f"wa_humidity_r{rd}")]
                raw[:, :, BASE.index(f"wa_r{rd}_humidity_gt100")] = np.where(np.isfinite(hum), hum > 100, np.nan)
                a, b, c = [raw[:, :, BASE.index(f"{v}_r{rd}")] for v in ("wa_temp", "wa_temp_max", "wa_temp_min")]
                raw[:, :, BASE.index(f"wa_r{rd}_temp_triplet_mismatch")] = np.where(np.isfinite(a) & np.isfinite(b) & np.isfinite(c), (a != b) | (a != c), np.nan)
    x = expand(raw, geometry(days, sites)); y = np.full((len(days), 24), np.nan, dtype="float64")
    if labels:
        if split != "train": raise ValueError("Test labels are unavailable")
        frame = pd.read_parquet(Path(root) / "train/generation.parquet", columns=["site_id", "ts", "energy"], filters=[("ts", ">=", ticks.min()), ("ts", "<=", ticks.max())])
        frame["ts"] = timestamps(frame.ts)
        if frame.duplicated(["site_id", "ts"]).any(): raise ValueError("Duplicate generation key")
        if ((frame.ts.dt.minute != 0) | (frame.ts.dt.second != 0) | (frame.ts.dt.microsecond != 0) | (frame.ts.dt.nanosecond != 0)).any(): raise ValueError("Off-hour generation requires review")
        if not frame.site_id.isin(load_sites(root).site_id).all(): raise ValueError("Generation references an unknown site")
        keys = pd.MultiIndex.from_arrays([np.repeat(days.site_id.to_numpy(), 24), ticks], names=["site_id", "ts"])
        energy = frame.set_index(["site_id", "ts"]).energy.reindex(keys).to_numpy().reshape(-1, 24)
        cap = days.site_id.map(meta.capacity_kw).to_numpy()
        good = np.isfinite(energy) & (energy >= 0)
        y[good] = (energy / cap[:, None])[good]
    return x, y, days, {"off_hour_forecasts_excluded": excluded, "labels_loaded": labels}
