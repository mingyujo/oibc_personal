"""Frozen last-year benchmark: lock the recipe before reading test outcomes."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
import time
import importlib.metadata
import sys
import numpy as np
import pandas as pd
from . import competition as c, data, evaluation, features as f, models
from .io import RunLock, event, fingerprint, private_root, read, sha, write
from .testbed_report import compare, render

VERSION = "frozen-last-year-testbed-v1"


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Invalid testbed/candidate name")
    return value


def specification(path):
    spec = read(path)
    keys = {"profile", "cache_start", "cache_end", "split", "seeds", "candidates", "baseline", "training", "statistics", "minimum_common_hours"}
    if set(spec)-keys or not {"profile", "split", "seeds", "candidates", "baseline"} <= set(spec):
        raise ValueError("Testbed configuration schema mismatch")
    if spec["profile"] not in {"competition", "synthetic"}: raise ValueError("Unknown testbed profile")
    split = spec["split"]
    if set(split) != {"cutoff", "test_start", "test_end"}: raise ValueError("One frozen test interval is required")
    dates = {key: f.stamp(value) for key, value in split.items()}
    if any(value != value.normalize() for value in dates.values()) or not dates["cutoff"] < dates["test_start"] <= dates["test_end"]:
        raise ValueError("Invalid target-day split")
    if spec["profile"] == "competition" and split != {"cutoff": c.CUTOFF, "test_start": c.TEST_START, "test_end": c.TEST_END}:
        raise ValueError("Competition test is frozen to2024-06-01..2025-05-31; cutoff2024-05-31")
    if not isinstance(spec["seeds"], list) or not spec["seeds"] or any(type(s) is not int or not 0 <= s < 2**31 for s in spec["seeds"]) or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Seeds must be unique nonnegative integers")
    if not isinstance(spec["candidates"], list) or not spec["candidates"]: raise ValueError("No candidates")
    names = []
    for candidate in spec["candidates"]:
        if set(candidate)-{"id", "model", "recipe", "composition"} or not {"id", "model"} <= set(candidate):
            raise ValueError("Candidate configuration schema mismatch")
        names.append(identifier(candidate["id"]))
        if candidate["model"] not in (*models.MODEL_NAMES, "zero"): raise ValueError("Unknown candidate model")
        candidate.setdefault("recipe", "raw6"); f.recipe_indices(candidate["recipe"])
        candidate.setdefault("composition", "reference")
        policy = candidate["composition"]
        if policy != "reference":
            if not isinstance(policy, dict) or set(policy) != {"file", "fitted_through"}:
                raise ValueError("Custom composition needs file and fitted_through")
            through = pd.Timestamp(policy["fitted_through"])
            if through.tz is None or through.tz_convert(f.TZ) > dates["cutoff"]:
                raise ValueError("Composition must be frozen before the training cutoff")
    if len(set(names)) != len(names) or spec["baseline"] not in names: raise ValueError("Duplicate candidate or missing baseline")
    training = spec.setdefault("training", {})
    valid_training = set(models.Config.__dataclass_fields__)-{"model", "recipe", "cutoff", "seed"}
    if set(training)-valid_training: raise ValueError("Unknown training option")
    cfg = models.Config(**training)
    if min(cfg.threads, cfg.max_rounds, cfg.patience, cfg.validation_days, cfg.epochs, cfg.neural_patience, cfg.batch_size) < 1:
        raise ValueError("Training counts must be positive")
    if cfg.device not in {"cpu", "gpu"} or cfg.history_days is not None and cfg.history_days <= cfg.validation_days+1:
        raise ValueError("Invalid training device/history window")
    stats = spec.setdefault("statistics", {"replicates": 1000, "block_days": 7})
    if set(stats) != {"replicates", "block_days"} or any(type(v) is not int or v < 1 for v in stats.values()):
        raise ValueError("Invalid bootstrap configuration")
    spec.setdefault("minimum_common_hours", 48)
    if type(spec["minimum_common_hours"]) is not int or spec["minimum_common_hours"] < 1:
        raise ValueError("Invalid common-support requirement")
    spec.setdefault("cache_start", "2021-01-01" if spec["profile"] == "competition" else "2024-01-01")
    spec.setdefault("cache_end", split["test_end"])
    if f.stamp(spec["cache_start"]) >= dates["cutoff"] or f.stamp(spec["cache_end"]) < dates["test_end"]:
        raise ValueError("Cache must contain training and the entire test interval")
    return spec


def plan(config):
    spec = specification(config)
    unique = {(candidate["model"], candidate["recipe"]) for candidate in spec["candidates"] if candidate["model"] != "zero"}
    count = len(unique)*len(spec["seeds"])
    return {"version": VERSION, "profile": spec["profile"], "split": spec["split"], "candidates": len(spec["candidates"]),
            "seeds": len(spec["seeds"]), "unique_models": count, "maximum_fit_stages": 2*count,
            "scoring_jobs": len(spec["candidates"])*len(spec["seeds"]), "objective": "S=R-W on identical complete-site hours",
            "all_test_results_are_development": True, "training_started": False}


def read_prices(root, split):
    prices = c.price_series(Path(root)/"train/smp.csv")
    times = pd.date_range(f.stamp(split["test_start"])+pd.Timedelta("1h"), f.stamp(split["test_end"])+pd.Timedelta("1d"), freq="h")
    covered = prices.reindex(times).to_numpy()
    if not np.isfinite(covered).all(): raise ValueError("Training SMP must cover the entire test year")
    return prices


def last_target_day(root):
    import pyarrow.parquet as pq
    path = Path(root)/"train/generation.parquet"; parquet = pq.ParquetFile(path)
    column = parquet.schema_arrow.get_field_index("ts"); maximum = []
    for index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(index).column(column).statistics
        if stats is None or not stats.has_min_max:
            times = pd.read_parquet(path,columns=["ts"]).ts
            return (f.timestamps(times).max()-pd.Timedelta("1ns")).normalize()
        maximum.append(f.stamp(stats.max))
    if not maximum: raise ValueError("Generation history is empty")
    return (max(maximum)-pd.Timedelta("1ns")).normalize()


def complete_cohort(work, split, sites):
    counts = pd.Series(0, index=sites.site_id, dtype="int64")
    for _, y, days in data.chunks(work, split["test_start"], split["test_end"]):
        values = pd.Series(np.isfinite(y).sum(axis=1), index=days.site_id).groupby(level=0).sum()
        counts = counts.add(values, fill_value=0)
    hours = (f.stamp(split["test_end"])-f.stamp(split["test_start"])).days*24+24
    return set(counts.index[counts == hours])


def score_month(pred, y, days, sites, group, prices):
    dates = pd.DatetimeIndex(sorted(days.day.unique())); di = dates.get_indexer(days.day)
    si = pd.Index(sites.site_id).get_indexer(days.site_id)
    if (si < 0).any() or days.duplicated(["site_id", "day"]).any(): raise ValueError("Invalid evaluation keys")
    target_shape = (len(dates)*24, len(sites))
    truth = np.full(target_shape, np.nan); forecast = np.zeros(target_shape)
    ti = (di[:, None]*24+np.arange(24)).reshape(-1); plant = np.repeat(si, 24)
    cap = sites.capacity_kw.to_numpy()[si, None]
    truth[ti, plant] = (y*cap).reshape(-1); forecast[ti, plant] = (pred*cap).reshape(-1)
    times = pd.to_datetime(np.repeat(dates.as_unit("ns").asi8, 24)+np.tile(np.arange(1,25), len(dates))*pd.Timedelta("1h").value, utc=True).tz_convert(f.TZ)
    money, valid, revenue, penalty = c.settlement(forecast, truth, times, group, prices.reindex(times).to_numpy())
    daily = [{"day": day.strftime("%Y-%m-%d"), "R": float(revenue[i*24:(i+1)*24].sum()),
              "W": float(penalty[i*24:(i+1)*24].sum()), "common_hours": int(valid[i*24:(i+1)*24].sum())} for i,day in enumerate(dates)]
    return money, daily


def score(work, split, sites, group, cohort, prices, candidate, seed, model_dir, device):
    net = config = None
    if model_dir is not None: net, config, _ = models.load(model_dir, device)
    total, day_total, night_total, complete_total = [evaluation.Totals() for _ in range(4)]
    monthly, ledger = [], {}
    for x, y, days in data.chunks(work, split["test_start"], split["test_end"]):
        pred = np.zeros_like(y) if net is None else models.predict(net, config, x, model_dir, device)
        cap = days.site_id.map(sites.set_index("site_id").capacity_kw).to_numpy()
        total.add(pred,y,cap); light = x[:,:,212] > 0
        day_total.add(pred,np.where(light,y,np.nan),cap); night_total.add(pred,np.where(~light,y,np.nan),cap)
        mask = days.site_id.isin(cohort).to_numpy(); complete_total.add(pred[mask], y[mask], cap[mask])
        part = evaluation.Totals(); part.add(pred,y,cap)
        money, daily = score_month(pred,y,days,sites,group,prices)
        monthly.append({"month": days.day.iloc[0].strftime("%Y-%m"), **part.result(), **money})
        for row in daily:
            if row["day"] in ledger: raise ValueError("Duplicate evaluation calendar day")
            ledger[row["day"]] = row
    if not total.labels: raise ValueError("Test interval has no observed targets")
    keys = pd.date_range(split["test_start"], split["test_end"], freq="D").strftime("%Y-%m-%d").tolist()
    daily = [ledger.get(key, {"R":0., "W":0., "common_hours":0}) for key in keys]
    common = sum(r["common_hours"] for r in daily)
    revenue, penalty = sum(r["R"] for r in daily), sum(r["W"] for r in daily)
    existing = {row["month"]:row for row in monthly}
    months = pd.period_range(f.stamp(split["test_start"]).tz_localize(None),f.stamp(split["test_end"]).tz_localize(None),freq="M")
    monthly = [existing.get(str(month),{"month":str(month),**evaluation.Totals().result(),"common_hours":0,"R":None,"W":None,"S":None}) for month in months]
    return {"candidate":candidate["id"], "seed":seed, "model":candidate["model"], "recipe":candidate["recipe"],
            "all":total.result(), "day":day_total.result(), "night":night_total.result(), "annual_complete":complete_total.result(),
            "complete_cohort_sites":len(cohort), "settlement":{"common_hours":common, "R":revenue if common else None,
                "W":penalty if common else None, "S":revenue-penalty if common else None},
            "monthly":monthly, "daily_keys":keys, "support_hours":[r["common_hours"] for r in daily],
            "daily_R":[r["R"] for r in daily], "daily_W":[r["W"] for r in daily]}


def run(root, work, config_file, name="last-year", resume=False):
    spec = specification(config_file); identifier(name); root = Path(root).resolve(); work = private_root(work)
    sites = f.load_sites(root); c.validate_sites(sites,spec["profile"])
    if spec["profile"] == "competition" and last_target_day(root) != f.stamp(c.TEST_END):
        raise ValueError("Provided generation does not end on the frozen last-year boundary")
    policies, composition_hashes = {}, {}
    for candidate in spec["candidates"]:
        incoming = None; policy = candidate["composition"]
        if policy != "reference":
            path = Path(policy["file"])
            if not path.is_absolute(): path = Path(config_file).resolve().parent/path
            incoming = pd.read_csv(path); composition_hashes[candidate["id"]] = {"path":str(path.resolve()), "sha256":sha(path), "fitted_through_is_user_declaration":True}
        group, capacities = c.composition(sites,incoming)
        policies[candidate["id"]] = group
    folder = work/"testbeds"/name
    if folder.exists() and not resume: raise FileExistsError("Testbed exists; inspect it and use --resume or another run name")
    folder.mkdir(parents=True,exist_ok=True)
    with RunLock(folder,resume):
        event("testbed_preflight", **plan(config_file))
        prices = read_prices(root,spec["split"])
        data.prepare(root,work,spec["cache_start"],spec["cache_end"],resume)
        cache, manifest = data.cache_info(work)
        implementation = fingerprint({p.name:sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        environment = {p:importlib.metadata.version(p) for p in ("numpy","pandas","pyarrow","lightgbm","catboost")}
        if any(x["model"] in ("mlp24","patch3") for x in spec["candidates"]): environment["torch"] = importlib.metadata.version("torch")
        contract = {"version":VERSION, "specification":spec, "cache_sha256":sha(cache/"manifest.json"),
                    "price_sha256":sha(root/"train/smp.csv"), "composition_sources":composition_hashes,
                    "implementation":implementation, "python":list(sys.version_info[:2]), "packages":environment}
        contract_path = folder/"contract.json"
        if contract_path.exists() and read(contract_path) != contract: raise ValueError("Frozen testbed contract changed; use a new run name")
        if not contract_path.exists(): write(contract_path,contract)
        # This record is frozen before test target/cohort inspection or any scoring.
        state_path = folder/"state.json"; state = read(state_path) if state_path.exists() else {"complete":False,"jobs":{}}
        if state["complete"] and (sha(folder/"summary.json") != state["summary_sha256"] or sha(folder/"report.html") != state["report_sha256"]):
            raise ValueError("Completed testbed summary/report changed")
        cohort = complete_cohort(work,spec["split"],sites); results = []; trained = set()
        for candidate in spec["candidates"]:
            for seed in spec["seeds"]:
                job_id = candidate["id"]+"-"+str(seed); saved = state["jobs"].get(job_id)
                result_path = folder/(job_id+".json")
                if saved:
                    if sha(result_path) != saved["result_sha256"] or any(sha(work/p) != digest for p,digest in saved["model_files"].items()):
                        raise ValueError("Completed testbed artifact changed")
                    results.append(read(result_path)); continue
                begin = time.monotonic(); model_dir = None; model_files = {}
                if candidate["model"] != "zero":
                    cfg = models.Config(model=candidate["model"],recipe=candidate["recipe"],cutoff=spec["split"]["cutoff"],seed=seed,**spec["training"])
                    model_name = "tb-"+fingerprint({"name":name,"contract":contract,"config":asdict(cfg)})[:24]
                    model_dir = models.train(work,cfg,model_name,resume=resume or model_name in trained); trained.add(model_name)
                    paths = [model_dir/"contract.json",model_dir/"complete.json",models.artifact(model_dir,cfg)]
                    if cfg.model in ("mlp24","patch3"): paths.append(model_dir/"refit-scaler.npz")
                    model_files = {str(p.relative_to(work)):sha(p) for p in paths}
                result = score(work,spec["split"],sites,policies[candidate["id"]],cohort,prices,candidate,seed,model_dir,spec["training"].get("device","cpu"))
                result["seconds"] = time.monotonic()-begin
                write(result_path,result); results.append(result)
                state["jobs"][job_id] = {"result_sha256":sha(result_path),"model_files":model_files}
                write(state_path,state); event("testbed_job_complete",candidate=candidate["id"],seed=seed,common_hours=result["settlement"]["common_hours"])
        if data.sources(root) != manifest["contract"]["source_hashes"] or sha(root/"train/smp.csv") != contract["price_sha256"]:
            raise RuntimeError("Original benchmark inputs changed")
        if any(sha(v["path"]) != v["sha256"] for v in composition_hashes.values()): raise RuntimeError("Composition file changed")
        comparison = compare(results,spec["baseline"],spec["seeds"],spec["minimum_common_hours"],spec["statistics"]["replicates"],spec["statistics"]["block_days"])
        summary = {"version":VERSION,"profile":spec["profile"],"split":spec["split"],"baseline":spec["baseline"],
                   "statistics":spec["statistics"],"complete_cohort_sites":len(cohort),"comparison":comparison,"results":results,
                   "development_evaluation":True,"independent_final_test":False,"test_interval_frozen":True,
                   "test_year_frozen":spec["profile"] == "competition",
                   "automatic_promotion":False,"external_submission":False,"contract_sha256":sha(contract_path)}
        output = render(folder,summary)
        state.update(complete=True,summary_sha256=sha(folder/"summary.json"),report_sha256=sha(output)); write(state_path,state)
        event("testbed_complete",jobs=len(results),report=str(output),profile=spec["profile"])
    return folder


def report(work,name):
    folder = private_root(work)/"testbeds"/identifier(name)
    state = read(folder/"state.json")
    if not state["complete"] or sha(folder/"summary.json") != state["summary_sha256"]:
        raise ValueError("Complete, unmodified testbed results required")
    return render(folder,read(folder/"summary.json"))


def demo(work):
    from .demo import make_dataset
    work = private_root(work); name = "demo-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    root = make_dataset(work/"demo-inputs"/name,48,official=True)
    spec = {"profile":"synthetic","cache_start":"2024-01-01","cache_end":"2024-02-17",
            "split":{"cutoff":"2024-02-05","test_start":"2024-02-06","test_end":"2024-02-17"},"seeds":[20261024],
            "baseline":"zero","candidates":[{"id":"zero","model":"zero"},{"id":"lgbm_raw6","model":"lgbm_mae"}],
            "training":{"threads":2,"max_rounds":5,"patience":2,"validation_days":5},"statistics":{"replicates":100,"block_days":3},"minimum_common_hours":24}
    config = work/"demo-inputs"/name/"testbed.json"; write(config,spec)
    folder = run(root,work/"demo-runs"/name,config,name)
    inner_work = private_root(work/"demo-runs"/name)
    checkpoints = list((inner_work/"models").iterdir())
    output = evaluation.predict_submission(root,inner_work,checkpoints[0])
    checked = c.validate_submission(root,output/"composition.csv",output/"forecast.csv",profile="synthetic")
    write(folder/"demo-submission-check.json",checked)
    event("testbed_demo_complete",report=str(folder/"report.html"),submission_valid=checked["valid"],synthetic=True)
    return folder
