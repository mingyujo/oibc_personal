"""Score supplied CSV predictions against the frozen historical year, without fits."""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import hashlib
import json
import re
import threading
import numpy as np
import pandas as pd
from . import competition as c, features as f
from .display import won
from .io import private_root, read, sha, write, fingerprint
from .local_details import write_csv

VERSION = "csv-last-year-testbed-v1"


def label(value):
    value = str(value).strip()
    if not value or len(value)>100 or re.search(r"\b(?:PV|WA|WB)\d{3}\b",value):
        raise ValueError("Use a model/experiment label without original source identifiers")
    return value


class Truth:
    def __init__(self,root,profile="competition",start=c.TEST_START,end=c.TEST_END):
        self.root=Path(root).resolve(); self.profile=profile
        if profile=="competition" and (start,end)!=(c.TEST_START,c.TEST_END):
            raise ValueError("Last-year test dates are frozen")
        if profile=="competition":
            from .testbed import last_target_day
            if last_target_day(self.root)!=f.stamp(end): raise ValueError("Dataset does not end at the prescribed last-year boundary")
        self.start,self.end=start,end
        files=[self.root/name for name in ("sites.csv","train/generation.parquet","train/smp.csv")]
        before=[sha(path) for path in files]
        self.sites=f.load_sites(self.root); c.validate_sites(self.sites,profile)
        self.times=pd.date_range(f.stamp(start)+pd.Timedelta("1h"),f.stamp(end)+pd.Timedelta("1d"),freq="h")
        frame=pd.read_parquet(files[1],columns=["site_id","ts","energy"],filters=[("ts",">=",self.times[0]),("ts","<=",self.times[-1])])
        frame["ts"]=c.ticks(frame.ts)
        if frame.duplicated(["site_id","ts"]).any() or not frame.site_id.isin(self.sites.site_id).all():
            raise ValueError("Invalid truth keys")
        values=pd.to_numeric(frame.energy,errors="raise")
        frame["energy"]=values.where(np.isfinite(values)&values.ge(0))
        self.y=frame.pivot(index="ts",columns="site_id",values="energy").reindex(index=self.times,columns=self.sites.site_id).to_numpy(float)
        self.common=np.isfinite(self.y).all(axis=1)
        self.prices=c.price_series(files[2]).reindex(self.times).to_numpy()
        if not np.isfinite(self.prices).all(): raise ValueError("Price coverage is incomplete")
        if before!=[sha(path) for path in files]: raise RuntimeError("Dataset changed while loading")
        self.hashes=dict(zip(("sites","generation","prices"),before))
        self.files=files; self.signatures=[(p.stat().st_size,p.stat().st_mtime_ns) for p in files]
        self.context={"version":VERSION,"profile":profile,"test_start":start,"test_end":end,"hours":len(self.times),
                      "sites":len(self.sites),"complete_cohort_sites":int(np.isfinite(self.y).all(axis=0).sum()),
                      "common_hours":int(self.common.sum()),"source_hashes":self.hashes,
                      "implementation":fingerprint({p.name:sha(p) for p in (Path(__file__),Path(c.__file__),Path(f.__file__))}),
                      "numpy":np.__version__,"pandas":pd.__version__}
        self.context_id=fingerprint(self.context)

    def verify(self):
        if [(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files]!=self.signatures:
            raise RuntimeError("Dataset changed; restart the CSV testbed")

    def score(self,forecast,composition=None,trained_through=None):
        self.verify()
        group,capacities=c.composition(self.sites,composition)
        aliases=[name for name in ("forecast_kwh","pred_kwh","prediction_kwh") if name in forecast]
        if len(aliases)!=1: raise ValueError("One explicit kWh prediction column is required")
        key="group" if "group" in forecast else "site_id" if "site_id" in forecast else None
        if key is None or set(forecast)!={key,"ts",aliases[0]}: raise ValueError("CSV must contain group or site_id, ts and forecast_kwh")
        forecast=forecast.rename(columns={aliases[0]:"forecast_kwh"}).copy()
        if forecast.isna().any().any(): raise ValueError("Forecast has missing fields")
        forecast["ts"]=c.ticks(forecast.ts)
        values=pd.to_numeric(forecast.forecast_kwh,errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any() or forecast.duplicated([key,"ts"]).any():
            raise ValueError("Forecast must be finite, nonnegative and unique")
        forecast["forecast_kwh"]=values
        if not forecast.ts.isin(self.times).all(): raise ValueError("Forecast contains timestamps outside the frozen test year")
        expected=["G1","G2"] if key=="group" else self.sites.site_id.tolist()
        if set(forecast[key])!=set(expected): raise ValueError("Forecast group/site coverage mismatch")
        matrix=forecast.pivot(index="ts",columns=key,values="forecast_kwh").reindex(index=self.times,columns=expected).to_numpy(float)
        # All candidates must cover every scorable hour; none can drop a difficult observed hour.
        if not np.isfinite(matrix[self.common]).all(): raise ValueError("CSV omits one or more scorable last-year hours")
        full=bool(np.isfinite(matrix).all())
        site_metrics=None
        if key=="site_id":
            if full:
                good=np.isfinite(self.y); errors=matrix-self.y; cap=self.sites.capacity_kw.to_numpy()
                cohort=np.isfinite(self.y).all(axis=0)
                site_metrics={"all_observed_E":float(abs(errors[good]).sum()/np.broadcast_to(cap,self.y.shape)[good].sum()) if good.any() else None,
                              "complete_cohort_E":float(abs(errors[:,cohort]).sum()/(len(self.times)*cap[cohort].sum())) if cohort.any() else None}
            pred=np.column_stack([matrix[:,group==g].sum(axis=1) for g in ("G1","G2")])
        else: pred=matrix
        # Missing, unscorable forecast rows do not turn missing truths into zero-valued labels.
        pred=np.where(np.isnan(pred),0.,pred)
        actual=np.column_stack([np.nansum(self.y[:,group==g],axis=1) for g in ("G1","G2")])
        actual[~self.common]=np.nan
        money,valid,revenue,penalty=c.settlement(pred,actual,self.times,["G1","G2"],self.prices)
        status="unverified"; declared=None
        if trained_through:
            declared=f.stamp(trained_through)
            if declared>f.stamp(c.CUTOFF): raise ValueError("Declared model training uses test-period information")
            status="user_declared_pre_test_cutoff"
        monthly=[]
        # Target-day month assigns00:00 back to the preceding target day.
        months=(self.times-pd.Timedelta("1ns")).strftime("%Y-%m")
        for month in sorted(set(months)):
            take=months==month; count=int(valid[take].sum()); R=float(revenue[take].sum()); W=float(penalty[take].sum())
            monthly.append({"month":month,"common_hours":count,"R":R if count else None,"W":W if count else None,"S":R-W if count else None})
        self.verify()
        return {"version":VERSION,"prediction_level":key,"full_test_grid":full,"provided_rows":len(forecast),
                "training_time_check":status,"trained_through":declared.isoformat() if declared is not None else None,
                "group_capacity_kw":capacities,"settlement":money,"site_metrics":site_metrics,"monthly":monthly,
                "context_id":self.context_id,"development_evaluation":True,"independent_final_test":False,
                "external_submission":False,"raw_precision_preserved":True}


class Store:
    def __init__(self,truth,work):
        self.truth=truth; self.work=private_root(work); self.folder=self.work/"csv-testbed"
        self.folder.mkdir(parents=True,exist_ok=True); self.lock=threading.RLock()
        self.index_path=self.folder/"index.json"
        if self.index_path.exists():
            if read(self.index_path)["context_id"]!=truth.context_id: raise ValueError("CSV scoreboard belongs to a different dataset/test interval")
        else: write(self.index_path,{"context_id":truth.context_id,"context":truth.context,"results":{}})

    def add(self,forecast,composition=None,name="CSV result",trained_through=None,source_bytes=None):
        name=label(name)
        with self.lock:
            result=self.truth.score(forecast,composition,trained_through)
            token=fingerprint({"csv":hashlib.sha256(source_bytes).hexdigest() if source_bytes is not None else fingerprint(forecast.astype(str).to_dict("list")),
                               "groups":fingerprint(composition.astype(str).to_dict("list")) if composition is not None else "reference",
                               "label":name,"context":self.truth.context_id,"trained_through":result["trained_through"]})[:24]
            path=self.folder/(token+".json"); index=read(self.index_path)
            if token in index["results"]:
                if sha(path)!=index["results"][token]["sha256"]: raise ValueError("Saved CSV result changed")
                return read(path)
            forecast_path=write_csv(forecast,Path("csv-testbed")/token/"forecast.csv",root=self.work,index=False)
            composition_path=write_csv(composition,Path("csv-testbed")/token/"composition.csv",root=self.work,index=False) if composition is not None else None
            result.update(id=token,label=name,created_utc=datetime.now(timezone.utc).isoformat(),
                          original_csv_sha256=hashlib.sha256(source_bytes).hexdigest() if source_bytes is not None else None,
                          forecast_sha256=sha(forecast_path),composition_sha256=sha(composition_path) if composition_path else None)
            write(path,result); index["results"][token]={"sha256":sha(path),"file":path.name}; write(self.index_path,index)
            return result

    def add_file(self,path,composition=None,name=None,trained_through=None):
        raw=Path(path).read_bytes(); before=hashlib.sha256(raw).hexdigest()
        frame=pd.read_csv(StringIO(raw.decode("utf-8-sig")))
        comp=pd.read_csv(composition) if composition is not None else None
        result=self.add(frame,comp,name or "CSV-"+before[:10],trained_through,raw)
        if sha(path)!=before: raise RuntimeError("Input CSV changed while scoring")
        return result

    def results(self):
        self.truth.verify(); index=read(self.index_path); results=[]
        for item in index["results"].values():
            path=self.folder/item["file"]
            if sha(path)!=item["sha256"]: raise ValueError("Stored CSV score changed")
            result=read(path)
            result["display"]={name:won(result["settlement"][name]) for name in ("R","W","S")}
            for row in result["monthly"]: row["display"]={name:won(row[name]) for name in ("R","W","S")}
            results.append(result)
        from .research_archive import bundled_results
        local={(r["label"],r.get("forecast_sha256")) for r in results}
        results.extend(r for r in bundled_results(self.truth.context) if (r["label"],r.get("forecast_sha256")) not in local)
        for result in results:
            result["display"]={name:won(result["settlement"][name]) for name in ("R","W","S")}
            for row in result["monthly"]: row["display"]={name:won(row[name]) for name in ("R","W","S")}
        results.sort(key=lambda row:(row["settlement"]["S"] is None,-(row["settlement"]["S"] or 0),row["label"]))
        return results

    def export_scores(self):
        rows=[]
        for result in self.results():
            name=result["label"]
            if name.startswith(("=","+","-","@")): name="'"+name
            rows.append({"label":name,**{key+"_KRW":result["display"][key].replace(",","") if result["settlement"][key] is not None else "" for key in ("W","S","R")},
                         "common_hours":result["settlement"]["common_hours"],"training_time_check":result["training_time_check"]})
        return write_csv(pd.DataFrame(rows,columns=["label","W_KRW","S_KRW","R_KRW","common_hours","training_time_check"]),
                         Path("csv-testbed")/("scoreboard-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")+".csv"),root=self.work,index=False)
