"""Invented sites and weather for integration tests; no competition records."""
from pathlib import Path
import numpy as np
import pandas as pd
from oibc_portable import features as f


def make_dataset(root, train_days=48, official=False):
    root = Path(root); (root / "train").mkdir(parents=True, exist_ok=True); (root / "test").mkdir(exist_ok=True)
    sites = pd.DataFrame({"site_id": ["SYN-LEFT", "SYN-RIGHT"], "current_group": ["A", "B"], "capacity_kw": [21000., 22000.],
                          "lat": [34.2, 36.8], "lon": [127.1, 128.4], "wa_station": ["SYN-A1", "SYN-A2"],
                          "wb_station": ["SYN-B1", "SYN-B2"], "wa_dist": [1500., 2500.], "wb_dist": [2200., 3300.]})
    # CSV fixtures are invented, and are written through the same private export helper.
    from oibc_portable.local_details import write_csv
    private = root.parent / ".local-only"
    import shutil
    temp = write_csv(sites, "fixtures-sites.csv", root=private, index=False); shutil.copyfile(temp, root / "sites.csv")
    times = pd.date_range("2024-01-01 01:00", periods=train_days*24, freq="h", tz=f.TZ)
    all_times = {"train": times}
    if official: all_times["test"] = pd.date_range("2025-06-01 01:00", periods=8760, freq="h", tz=f.TZ)
    else: all_times["test"] = times[:24]
    for split, ticks in all_times.items():
        for provider in f.WEATHER:
            pieces = []
            for index, station in enumerate(sites[provider + "_station"]):
                for rd in (10, 17):
                    phase = np.arange(len(ticks))
                    rows = {provider + "_station": station, "round": rd, "ts": ticks}
                    for j, variable in enumerate(f.WEATHER[provider]):
                        rows[variable] = (30 + (phase % 19) + index*3 + j*.7 + (rd==10)*2).astype(float)
                    pieces.append(pd.DataFrame(rows))
            frame = pd.concat(pieces, ignore_index=True)
            frame.to_parquet(root / split / f"forecast_{provider}.parquet", index=False)
    pieces = []
    for index, site in sites.iterrows():
        ticks = times
        sun, _ = f.solar_terms(ticks - pd.Timedelta("30min"), np.full(len(ticks), site.lat), np.full(len(ticks), site.lon))
        energy = np.maximum(sun, 0) * (.7 - .002*(np.arange(len(ticks)) % 19)) * site.capacity_kw
        pieces.append(pd.DataFrame({"site_id": site.site_id, "ts": ticks, "energy": energy}))
    generation = pd.concat(pieces, ignore_index=True).drop(index=[5, 27]).reset_index(drop=True)
    generation.loc[0, "energy"] = -1.  # Excluded, never filled with zero.
    generation.to_parquet(root / "train/generation.parquet", index=False)
    prices = pd.DataFrame({"ts": times, "smp": 100.})
    temp = write_csv(prices, "fixtures-train-smp.csv", root=private, index=False); shutil.copyfile(temp, root / "train/smp.csv")
    prices = pd.DataFrame({"ts": all_times["test"], "smp": 100.})
    prices["k"] = np.where(prices.ts.dt.hour.isin([11,12,13,14,15]), 3, 1)
    temp = write_csv(prices, "fixtures-test-smp.csv", root=private, index=False); shutil.copyfile(temp, root / "smp.csv")
    return root
