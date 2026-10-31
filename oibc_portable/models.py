"""Two-stage fitting with an explicit embargo and clipped-output selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import gc
import importlib.metadata
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from . import RECIPE_VERSION
from . import data
from . import features as f
from .io import RunLock, Heartbeat, event, fingerprint, private_root, read, sha, write

MODEL_NAMES = ("lgbm_mae", "catboost_mae", "catboost_rmse", "mlp24", "patch3")


@dataclass
class Config:
    model: str = "lgbm_mae"
    recipe: str = "raw6"
    cutoff: str = "2024-05-31"
    seed: int = 20261024
    threads: int = 4
    device: str = "cpu"
    max_rounds: int = 6000
    patience: int = 300
    validation_days: int = 30
    history_days: int | None = None
    epochs: int = 50
    neural_patience: int = 5
    batch_size: int = 128


def masks(days, cutoff, validation_days=30):
    cutoff = f.stamp(cutoff).normalize(); first = cutoff - pd.Timedelta(days=validation_days)
    dates = pd.DatetimeIndex(days)
    train = np.asarray(dates + pd.Timedelta("1d") <= first - pd.Timedelta("1d"))
    valid = np.asarray((dates >= first) & (dates + pd.Timedelta("1d") <= cutoff))
    return train, valid


def clipped_mae(pred, truth, weight):
    good = np.isfinite(truth)
    if not good.any(): raise ValueError("No valid evaluation labels")
    return float(np.average(abs(np.maximum(pred[good], 0) - truth[good]), weights=weight[good]))


class ClippedMAE:
    def is_max_optimal(self): return False
    def evaluate(self, approxes, target, weight):
        total = 0.; exposure = 0.
        for i in range(len(target)):
            w = 1. if weight is None else weight[i]
            total += w * abs(max(approxes[0][i], 0.) - target[i]); exposure += w
        return total, exposure
    def get_final_error(self, error, weight): return error / max(weight, 1e-30)


def artifact(folder, config, stage="refit"):
    return Path(folder) / (stage + (".txt" if config.model == "lgbm_mae" else ".cbm" if config.model.startswith("catboost") else ".pt"))


def fit_tree(folder, config, x, y, cap, inner, valid):
    good = np.isfinite(y).reshape(-1)
    xx = x.reshape(-1, x.shape[-1])[good]; yy = y.reshape(-1)[good]
    weights = np.repeat(cap, 24)[good]; train = np.repeat(inner, 24)[good]; validation = np.repeat(valid, 24)[good]
    stages = []
    for stage in ("selection", "refit"):
        record = folder / (stage + ".json")
        if record.exists():
            info = read(record)
            if sha(artifact(folder, config, stage)) != info["model_sha256"]: raise ValueError("Saved stage model changed")
            stages.append(info); continue
        take = train if stage == "selection" else np.ones(len(yy), dtype=bool)
        maximum = config.max_rounds if stage == "selection" else stages[0]["iterations"]
        begin = time.monotonic()
        event("fit_stage_started", model=config.model, stage=stage, fit_labels=int(take.sum()), maximum_iterations=maximum)
        weighted = config.model != "catboost_rmse"
        training_weights = weights[take] / weights[take].mean() if weighted else None
        if config.model == "lgbm_mae":
            if config.device != "cpu": raise ValueError("Portable LightGBM recipe is CPU; use CatBoost or neural for GPU")
            import lightgbm as lgb
            params = dict(objective="regression_l1", metric="None", num_leaves=63, learning_rate=.03, min_data_in_leaf=128,
                          lambda_l2=5., feature_fraction=.9, bagging_fraction=.9, bagging_freq=1, num_threads=config.threads,
                          seed=config.seed, feature_fraction_seed=config.seed, bagging_seed=config.seed, data_random_seed=config.seed,
                          deterministic=True, force_col_wise=True, verbosity=-1)
            dataset = lgb.Dataset(xx[take], label=yy[take], weight=training_weights)
            def metric(p, d): return "clipped_capacity_MAE", clipped_mae(p, d.get_label(), d.get_weight()), False
            validation_sets = [lgb.Dataset(xx[validation], label=yy[validation], weight=weights[validation], reference=dataset)] if stage == "selection" else None
            callbacks = [lgb.early_stopping(config.patience, verbose=False)] if stage == "selection" else []
            def progress(env):
                if (env.iteration+1)%500==0: event("tree_iteration", model=config.model, stage=stage, iteration=env.iteration+1)
            with Heartbeat(config.model, stage):
                net = lgb.train(params, dataset, num_boost_round=maximum, valid_sets=validation_sets, feval=metric, callbacks=callbacks+[progress])
            iterations = net.best_iteration if stage == "selection" else maximum
        else:
            from catboost import CatBoostRegressor, Pool
            params = dict(loss_function="MAE" if config.model == "catboost_mae" else "RMSE", eval_metric=ClippedMAE(),
                          depth=8, learning_rate=.03, l2_leaf_reg=5., border_count=128, bootstrap_type="Bayesian", boosting_type="Plain",
                          random_seed=config.seed, thread_count=config.threads, allow_writing_files=False, iterations=maximum,
                          task_type="GPU" if config.device == "gpu" else "CPU")
            if config.device == "gpu": params.update(devices="0", gpu_ram_part=.75)
            net = CatBoostRegressor(**params)
            validation_pool = Pool(xx[validation], label=yy[validation], weight=weights[validation]) if stage == "selection" else None
            with Heartbeat(config.model, stage):
                net.fit(Pool(xx[take], label=yy[take], weight=training_weights), eval_set=validation_pool,
                        early_stopping_rounds=config.patience if stage == "selection" else None, use_best_model=stage == "selection", verbose=False)
            iterations = int(net.tree_count_)
        net.save_model(str(artifact(folder, config, stage)))
        info = {"stage": stage, "iterations": int(iterations), "seconds": time.monotonic() - begin,
                "fit_labels": int(take.sum()), "validation_labels": int(validation.sum()) if stage == "selection" else 0,
                "selection_metric": "clipped_capacity_MAE", "model_sha256": sha(artifact(folder, config, stage))}
        write(record, info); stages.append(info); event("fit_stage_complete", model=config.model, stage=stage, iterations=int(iterations))
        del net; gc.collect()
    del xx; gc.collect()
    return stages


def fit_neural(folder, config, x, y, cap, inner, valid):
    import torch
    from . import neural as n
    torch.set_num_threads(config.threads)
    device = "cuda:0" if config.device == "gpu" else "cpu"
    if config.device == "gpu" and not torch.cuda.is_available(): raise ValueError("GPU requested but PyTorch CUDA is unavailable")
    stages = []
    for stage in ("selection", "refit"):
        record = folder / (stage + ".json")
        if record.exists():
            info = read(record)
            if sha(artifact(folder, config, stage)) != info["model_sha256"]: raise ValueError("Saved stage model changed")
            stages.append(info); continue
        torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        take = inner if stage == "selection" else np.ones(len(y), dtype=bool)
        xx, yy = np.asarray(x[take]), np.asarray(y[take], dtype="float32")
        state = n.scaler(xx); np.savez(folder / (stage + "-scaler.npz"), **state)
        net = n.factory(config.model, x.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(config.seed); maximum = config.epochs if stage == "selection" else stages[0]["iterations"]
        best = float("inf"); best_epoch = 0; stale = 0; curve = []; begin = time.monotonic()
        for epoch in range(1, maximum + 1):
            net.train(); order = rng.permutation(len(xx))
            for start in range(0, len(order), config.batch_size):
                ix = order[start:start+config.batch_size]
                xb = torch.from_numpy(n.transform(xx[ix], state)).to(device); target = torch.from_numpy(yy[ix]).to(device); ok = torch.isfinite(target)
                optimizer.zero_grad(set_to_none=True); loss = (net(xb)[ok] - target[ok]).square().mean()
                if not torch.isfinite(loss): raise ValueError("Nonfinite neural loss")
                loss.backward(); optimizer.step()
            if stage == "selection":
                pred = n.predict(net, x[valid], state, device)
                score = clipped_mae(pred, y[valid], np.broadcast_to(cap[valid, None], y[valid].shape)); curve.append(score)
                if score < best:
                    best = score; best_epoch = epoch; stale = 0; torch.save(net.state_dict(), artifact(folder, config, stage))
                else: stale += 1
            event("neural_epoch_complete", model=config.model, stage=stage, epoch=epoch)
            if stage == "selection" and stale >= config.neural_patience: break
        if stage == "refit": torch.save(net.state_dict(), artifact(folder, config, stage))
        info = {"stage": stage, "iterations": best_epoch if stage == "selection" else maximum,
                "seconds": time.monotonic() - begin, "curve": curve, "model_sha256": sha(artifact(folder, config, stage)),
                "scaler_sha256": sha(folder / (stage + "-scaler.npz"))}
        write(record, info); stages.append(info); del net, optimizer, xx, yy; gc.collect()
    return stages


def train(work, config, run_name=None, resume=False):
    if config.model not in MODEL_NAMES or min(config.threads, config.max_rounds, config.epochs, config.validation_days, config.patience, config.neural_patience, config.batch_size) < 1: raise ValueError("Invalid model configuration")
    if config.device not in ("cpu", "gpu"): raise ValueError("Unknown device")
    if f.stamp(config.cutoff) != f.stamp(config.cutoff).normalize(): raise ValueError("Cutoff must be KST midnight")
    work = private_root(work); cache, manifest = data.cache_info(work)
    if config.history_days is not None and config.history_days <= config.validation_days + 1: raise ValueError("History window must include training, embargo and validation")
    run_name = run_name or f"{config.model}-{config.recipe}-{config.cutoff}-{config.seed}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in (".", ".."): raise ValueError("Invalid run name")
    folder = work / "models" / run_name; existed = folder.exists(); folder.mkdir(parents=True, exist_ok=True)
    if existed and not resume: raise FileExistsError("Run exists; use --resume or another --run-name")
    implementation = fingerprint({p.name: p.read_text(encoding="utf-8").replace("\r\n", "\n") for p in sorted(Path(__file__).parent.glob("*.py"))})
    package_names = ["numpy", "pandas", "pyarrow", "lightgbm", "catboost"]
    if config.model in ("mlp24", "patch3"): package_names.append("torch")
    environment = {p: importlib.metadata.version(p) for p in package_names}
    contract = {"version": RECIPE_VERSION, "config": asdict(config), "cache_sha256": sha(cache / "manifest.json"),
                "feature_contract": data.feature_contract(), "implementation": implementation,
                "python_major_minor": list(sys.version_info[:2]), "packages": environment,
                "source_hashes": manifest["contract"]["source_hashes"],
                "feature_indices": f.recipe_indices(config.recipe).tolist()}
    with RunLock(folder, resume):
        if (folder / "contract.json").exists():
            if read(folder / "contract.json") != contract: raise ValueError("Resume contract differs")
        else: write(folder / "contract.json", contract)
        if (folder / "complete.json").exists():
            if sha(artifact(folder, config)) != read(folder / "complete.json")["model_sha256"]: raise ValueError("Completed model changed")
            event("completed_run_reused", run=run_name); return folder
        event("training_data_preparation", run=run_name)
        x, y, days, cap = data.assemble(work, folder, config.cutoff, config.recipe, config.history_days)
        inner, valid = masks(days.day, config.cutoff, config.validation_days)
        if not inner.any() or not valid.any() or len(np.unique(days.day[valid])) != config.validation_days: raise ValueError("Insufficient full validation days or historical training data")
        first_origin = days.day[valid].min() - pd.Timedelta("1d") + pd.Timedelta("17h")
        latest_label = (days.day[inner] + pd.Timedelta("1d")).max()
        if latest_label > first_origin: raise ValueError("Inner training label availability violation")
        write(folder / "time-audit.json", {"validation_days": config.validation_days, "embargo_target_days": 1,
              "inner_origin_margin_hours": (first_origin-latest_label).total_seconds()/3600, "all_refit_label_ends_before_cutoff": True})
        if config.model in ("mlp24", "patch3"): stages = fit_neural(folder, config, x, y, cap, inner, valid)
        else: stages = fit_tree(folder, config, x, y, cap, inner, valid)
        packages = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost")}
        info = {"complete": True, "recipe_version": RECIPE_VERSION, "model_sha256": sha(artifact(folder, config)),
                "stages": stages, "training_days": len(days), "training_labels": int(np.isfinite(y).sum()),
                "features": x.shape[-1], "packages": packages, "target": "energy/capacity_kw", "output": "nonnegative normalized energy"}
        write(folder / "complete.json", info)
        del x, y; gc.collect()
        for name in ("working-X.npy", "working-y.npy"):
            path = (folder / name).resolve(); path.relative_to(folder.resolve()); path.unlink(missing_ok=True)
        event("training_complete", run=run_name, features=info["features"], selected_iterations=stages[0]["iterations"])
    return folder


def load(folder, device="cpu"):
    folder = Path(folder).resolve(); contract = read(folder / "contract.json"); config = Config(**contract["config"])
    complete = read(folder / "complete.json")
    if not complete["complete"] or contract["version"] != RECIPE_VERSION or contract["feature_contract"] != data.feature_contract() or sha(artifact(folder, config)) != complete["model_sha256"]: raise ValueError("Model integrity/version check failed")
    if config.model == "lgbm_mae":
        import lightgbm as lgb
        net = lgb.Booster(model_file=str(artifact(folder, config)))
    elif config.model.startswith("catboost"):
        from catboost import CatBoostRegressor
        net = CatBoostRegressor(); net.load_model(str(artifact(folder, config)))
    else:
        import torch
        from . import neural as n
        scaler_path = folder / "refit-scaler.npz"
        if sha(scaler_path) != complete["stages"][1]["scaler_sha256"]: raise ValueError("Model scaler changed")
        net = n.factory(config.model, complete["features"]); net.load_state_dict(torch.load(artifact(folder, config), weights_only=True, map_location="cpu")); net.to("cuda:0" if device == "gpu" else "cpu").eval()
    return net, config, contract


def predict(net, config, raw6, folder, device="cpu"):
    x = np.asarray(raw6)[:, :, f.recipe_indices(config.recipe)]
    if config.model == "lgbm_mae": p = net.predict(x.reshape(-1, x.shape[-1]), num_threads=config.threads).reshape(-1, 24)
    elif config.model.startswith("catboost"): p = net.predict(x.reshape(-1, x.shape[-1]), thread_count=config.threads).reshape(-1, 24)
    else:
        from . import neural as n
        p = n.predict(net, x, dict(np.load(Path(folder) / "refit-scaler.npz")), "cuda:0" if device == "gpu" else "cpu")
    if not np.isfinite(p).all(): raise ValueError("Nonfinite model predictions")
    return np.maximum(p, 0)
