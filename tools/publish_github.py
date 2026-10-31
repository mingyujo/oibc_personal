"""Optional local publishing with an already-authenticated GitHub CLI.

Only check_release.py's allowlisted files are staged. Existing remote branches
are never overwritten. This helper does not read, request or print credentials.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote
from check_release import ROOT, audit


def run(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", **kwargs)
    if result.returncode: raise RuntimeError("Command failed: " + " ".join(args[:2]))
    return result.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True,help="owner/repository")
    parser.add_argument("--create",action="store_true",help="Create a new private repository; fails if it already exists")
    parser.add_argument("--branch",help="A new remote branch; defaults to main for a new repository or a timestamped branch")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",args.repo): raise ValueError("Invalid repository name")
    result=audit()
    if not result["passed"]: raise ValueError("Release allowlist check failed")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch=args.branch or ("main" if args.create else "codex/oibc-portable-"+stamp)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+",branch) or ".." in branch or branch.endswith("/"): raise ValueError("Invalid branch")
    if args.dry_run:
        print(json.dumps({"dry_run":True,"repo":args.repo,"branch":branch,"files":len(result["files"]),"private_if_created":True})); return
    if not shutil.which("gh") or not shutil.which("git"): raise RuntimeError("Install Git and GitHub CLI, then run gh auth login")
    run(["gh","auth","status"])
    if not args.create:
        state=json.loads(run(["gh","api","repos/"+args.repo]))
        if not state["private"]: raise ValueError("This helper publishes only to private repositories")
        exists=subprocess.run(["gh","api",f"repos/{args.repo}/branches/{quote(branch,safe='')}"],capture_output=True)
        if exists.returncode==0: raise ValueError("Remote branch exists; choose a new branch")
    stage=ROOT/".local-only"/"release-git"/stamp
    stage.mkdir(parents=True,exist_ok=False)
    for name in result["files"]:
        destination=stage/name; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,destination)
    run(["git","init","-b",branch],cwd=stage)
    user=json.loads(run(["gh","api","user"]))
    run(["git","config","user.name",user["login"]],cwd=stage)
    run(["git","config","user.email",f'{user["id"]}+{user["login"]}@users.noreply.github.com'],cwd=stage)
    run(["git","add","--",*result["files"]],cwd=stage)
    run(["git","commit","-m","Add reviewed portable solar forecasting models"],cwd=stage)
    if args.create: run(["gh","repo","create",args.repo,"--private"])
    run(["git","remote","add","origin",f"https://github.com/{args.repo}.git"],cwd=stage)
    # Git obtains authentication through GitHub CLI's normal credential helper.
    run(["git","-c","credential.helper=","-c","credential.helper=!gh auth git-credential","push","origin",f"HEAD:refs/heads/{branch}"],cwd=stage)
    local=run(["git","rev-parse","HEAD"],cwd=stage).strip()
    remote=json.loads(run(["gh","api",f"repos/{args.repo}/branches/{quote(branch,safe='')}"]))["commit"]["sha"]
    if local!=remote: raise RuntimeError("Remote commit verification failed")
    print(json.dumps({"uploaded":True,"repo":args.repo,"branch":branch,"commit":remote,"files":len(result["files"])}))


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"uploaded":False,"error":str(exc)}),file=sys.stderr); raise SystemExit(1)
