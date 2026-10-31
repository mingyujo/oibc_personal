"""Command line entrypoint. Detailed errors remain in private logs."""
from __future__ import annotations
import argparse
from contextlib import redirect_stderr
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import traceback
from .io import event, private_root, read, write, set_event_file


def parser():
    p = argparse.ArgumentParser(prog="oibc", description="Portable local-data solar models; no source rows are printed")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--work-dir", default=".local-only")
    prep = sub.add_parser("prepare"); prep.add_argument("--data-dir", required=True); prep.add_argument("--work-dir", default=".local-only")
    prep.add_argument("--start"); prep.add_argument("--end"); prep.add_argument("--resume", action="store_true")
    train = sub.add_parser("train"); train.add_argument("--work-dir", default=".local-only"); train.add_argument("--cutoff", required=True)
    train.add_argument("--model", choices=["lgbm_mae", "catboost_mae", "catboost_rmse", "mlp24", "patch3"], default="lgbm_mae")
    train.add_argument("--recipe", choices=["raw6", "base137", "no_weather"], default="raw6")
    for name, default in [("seed",20261024),("threads",4),("max-rounds",6000),("patience",300),("validation-days",30),("epochs",50),("neural-patience",5),("batch-size",128)]:
        train.add_argument("--"+name, type=int, default=default)
    train.add_argument("--history-days", type=int); train.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    train.add_argument("--run-name"); train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate"); evaluate.add_argument("--work-dir", default=".local-only"); evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--start", required=True); evaluate.add_argument("--end", required=True); evaluate.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    predict = sub.add_parser("predict"); predict.add_argument("--data-dir", required=True); predict.add_argument("--work-dir", default=".local-only")
    predict.add_argument("--model-dir", required=True); predict.add_argument("--composition"); predict.add_argument("--device", choices=["cpu","gpu"], default="cpu")
    experiment = sub.add_parser("experiment"); experiment.add_argument("--data-dir", required=True); experiment.add_argument("--work-dir", default=".local-only")
    experiment.add_argument("--config", required=True); experiment.add_argument("--resume", action="store_true")
    bed = sub.add_parser("testbed", help="Frozen last-year benchmark and competition submission checks")
    actions = bed.add_subparsers(dest="action", required=True)
    for action in ("plan", "run", "report", "demo", "validate-submission", "score-csv", "serve"):
        command = actions.add_parser(action)
        command.add_argument("--work-dir", default=".local-only")
        if action in ("plan", "run"): command.add_argument("--config", required=True)
        if action in ("run", "validate-submission", "score-csv", "serve"): command.add_argument("--data-dir", required=True)
        if action in ("run", "report"): command.add_argument("--run-name", default="last-year")
        if action == "run": command.add_argument("--resume", action="store_true")
        if action == "serve": command.add_argument("--port",type=int,default=8765)
        if action == "score-csv":
            command.add_argument("--forecast",nargs="+",required=True);command.add_argument("--composition")
            command.add_argument("--label");command.add_argument("--trained-through")
        if action == "validate-submission":
            command.add_argument("--composition", required=True); command.add_argument("--forecast", required=True)
            command.add_argument("--profile", choices=["competition", "synthetic"], default="competition")
    return p


def dispatch(args):
    from . import data, models, evaluation
    if args.command == "doctor":
        import psutil
        versions = {}
        for pkg in ("numpy","pandas","pyarrow","lightgbm","catboost","psutil","torch"):
            try: versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError: versions[pkg] = None
        work = private_root(args.work_dir)
        event("environment", python=sys.version.split()[0], platform=sys.platform, packages=versions,
              available_RAM_GiB=psutil.virtual_memory().available/2**30, free_disk_GiB=shutil.disk_usage(work).free/2**30)
    elif args.command == "prepare": data.prepare(args.data_dir, args.work_dir, args.start, args.end, args.resume)
    elif args.command == "train":
        fields = {name: getattr(args, name) for name in models.Config.__dataclass_fields__}
        models.train(args.work_dir, models.Config(**fields), args.run_name, args.resume)
    elif args.command == "evaluate": evaluation.evaluate(args.work_dir, args.model_dir, args.start, args.end, args.device)
    elif args.command == "predict": evaluation.predict_submission(args.data_dir, args.work_dir, args.model_dir, args.device, args.composition)
    elif args.command == "testbed":
        from . import testbed, competition
        if args.action == "plan": event("testbed_plan", **testbed.plan(args.config))
        elif args.action == "run": testbed.run(args.data_dir,args.work_dir,args.config,args.run_name,args.resume)
        elif args.action == "report": event("testbed_report_ready",report=str(testbed.report(args.work_dir,args.run_name)))
        elif args.action == "demo": testbed.demo(args.work_dir)
        elif args.action == "serve":
            from .csv_server import serve
            if not 1<=args.port<=65535: raise ValueError("Invalid local port")
            serve(args.data_dir,args.work_dir,args.port)
        elif args.action == "score-csv":
            from .csv_testbed import Truth,Store
            from .display import won
            from .io import RunLock
            store=Store(Truth(args.data_dir),args.work_dir)
            with RunLock(store.folder,resume=True):
                for path in args.forecast:
                    row=store.add_file(path,args.composition,args.label if len(args.forecast)==1 else Path(path).stem,args.trained_through)
                    event("csv_scored",label=row["label"],W_KRW=won(row["settlement"]["W"]),S_KRW=won(row["settlement"]["S"]),common_hours=row["settlement"]["common_hours"])
                event("csv_scoreboard_ready",scoreboard=str(store.export_scores()))
        elif args.action == "validate-submission":
            checked = competition.validate_submission(args.data_dir,args.composition,args.forecast,args.profile)
            output = private_root(args.work_dir)/"submission-checks"/(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")+".json")
            write(output,checked); event("submission_checked",**checked,report=str(output))
    elif args.command == "experiment":
        spec = read(args.config)
        data.prepare(args.data_dir, args.work_dir, spec.get("cache_start"), spec.get("cache_end"), args.resume)
        work = private_root(args.work_dir); results = []
        for cutoff in spec["cutoffs"]:
            for name in spec["models"]:
                for recipe in spec["recipes"]:
                    for seed in spec["seeds"]:
                        cfg = models.Config(model=name, recipe=recipe, cutoff=cutoff, seed=seed, **spec.get("training", {}))
                        folder = models.train(work, cfg, resume=args.resume)
                        from .features import stamp
                        first = stamp(cutoff) + __import__("pandas").Timedelta("1d")
                        score = evaluation.evaluate(work, folder, first, spec["evaluation_end"], cfg.device)
                        results.append({"model": name, "recipe": recipe, "cutoff": cutoff, "seed": seed, "evaluation": str(score)})
                        write(work / "experiment-index.json", {"config": spec, "results": results, "complete": False})
        write(work / "experiment-index.json", {"config": spec, "results": results, "complete": True})
        event("experiment_complete", runs=len(results))


def main(argv=None):
    args = parser().parse_args(argv)
    # No global environment or system PATH changes; limits apply to this process.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(getattr(args, "threads", 4)))
    work = private_root(args.work_dir); logs = work / "logs"; logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    log = logs / (args.command + "-" + stamp + ".log")
    set_event_file(logs / (args.command + "-" + stamp + ".events.jsonl"))
    try:
        with log.open("w", encoding="utf-8") as stream, redirect_stderr(stream): dispatch(args)
    except Exception as exc:
        with log.open("a", encoding="utf-8") as stream: traceback.print_exc(file=stream)
        event("failed", error_type=type(exc).__name__, details_log=str(log))
        return 1
    return 0
