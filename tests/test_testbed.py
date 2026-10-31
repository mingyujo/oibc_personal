from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from oibc_portable import competition as c, testbed
from oibc_portable.io import private_root, read, sha, write
from oibc_portable.local_details import write_csv
from oibc_portable.testbed_report import compare
from synthetic_data import make_dataset


class SettlementContract(unittest.TestCase):
    def test_group_cancellation_and_exact_revenue_penalty(self):
        times = pd.date_range("2024-01-01 10:00",periods=2,freq="h",tz="Asia/Seoul")
        y = np.full((2,4),100.)
        p = y + np.array([[10,-10,10,-10],[10,10,-10,-10]])
        result, valid, _, penalty = c.settlement(p,y,times,["G1","G1","G2","G2"],[100,100])
        self.assertTrue(valid.all()); self.assertEqual(penalty.tolist(),[0,1200])
        self.assertEqual(result,{"common_hours":2,"R":80000.,"W":1200.,"S":78800.})

    def test_end_hour_weight_zero_price_and_midnight(self):
        times = c.ticks(["2024-01-01 10:00+09:00","2024-01-01 11:00+09:00","2024-01-01 15:00+09:00","2024-01-01 16:00+09:00","2024-01-02 00:00+09:00"])
        self.assertEqual(c.weights(times).tolist(),[1,3,3,1,1])
        result,*_ = c.settlement([[0,0]],[[100,100]],times[:1],["G1","G2"],[0])
        self.assertEqual(result,{"common_hours":1,"R":0.,"W":0.,"S":0.})
        self.assertEqual((times[-1]-pd.Timedelta("1ns")).day,1)

    def test_missing_truth_excludes_whole_hour_not_candidate_errors(self):
        t = pd.date_range("2024-01-01 11:00",periods=2,freq="h",tz="Asia/Seoul")
        p = np.ones((2,2)); y = np.array([[1,1],[np.nan,1]])
        result,valid,*_ = c.settlement(p,y,t,["G1","G2"],[100,100])
        self.assertEqual(valid.tolist(),[True,False]); self.assertEqual(result["common_hours"],1)
        y[:] = np.nan
        self.assertIsNone(c.settlement(p,y,t,["G1","G2"],[100,100])[0]["S"])
        for bad in (np.nan,np.inf,-1):
            q = p.copy(); q[0,0] = bad
            with self.assertRaises(ValueError): c.settlement(q,y,t,["G1","G2"],[100,100])
        with self.assertRaises(ValueError): c.settlement(p,y,t,["G1","G2"],[100,np.nan])

    def test_composition_capacity_and_complete_assignment(self):
        sites = pd.DataFrame({"site_id":["SYN-A","SYN-B","SYN-C","SYN-D"],"capacity_kw":[10000.]*4,"current_group":["A","A","B","B"]})
        c.validate_sites(sites,"synthetic")
        group,cap = c.composition(sites)
        self.assertEqual(cap,{"G1":20000.,"G2":20000.})
        valid = pd.DataFrame({"site_id":sites.site_id,"group":group})
        for bad in (valid.iloc[:-1],pd.concat([valid,valid.iloc[:1]]),valid.assign(group="G1"),valid.assign(extra=0)):
            with self.assertRaises(ValueError): c.composition(sites,bad)
        with self.assertRaises(ValueError): c.validate_sites(sites,"competition")
        with self.assertRaises(ValueError): c.ticks(["2024-01-01 01:00:00"])
        with self.assertRaises(ValueError): c.ticks(["2024-01-01 01:01:00+09:00"])
        mixed=c.ticks(["2024-01-01 01:00:00+09:00","2023-12-31 16:00:00Z"])
        self.assertEqual(mixed[0],mixed[1])

    def test_full_submission_validation_and_invalid_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",2); private=private_root(Path(tmp)/"out")
            t=pd.date_range("2025-06-01 01:00",periods=8760,freq="h",tz="Asia/Seoul")
            prices=pd.DataFrame({"ts":t,"smp":0.,"k":c.weights(t)})
            saved=write_csv(prices,"price.csv",root=private,index=False); shutil.copyfile(saved,root/"smp.csv")
            comp=pd.DataFrame({"site_id":["SYN-LEFT","SYN-RIGHT"],"group":["G1","G2"]})
            cp=write_csv(comp,"composition.csv",root=private,index=False)
            forecast=pd.DataFrame({"group":np.repeat(["G1","G2"],len(t)),"ts":list(t)*2,"forecast_kwh":0.})
            fp=write_csv(forecast,"forecast.csv",root=private,index=False)
            self.assertTrue(c.validate_submission(root,cp,fp,"synthetic")["valid"])
            for kind in ("negative","duplicate","naive","missing","unknown","infinity"):
                bad=forecast.copy()
                if kind=="negative": bad.loc[0,"forecast_kwh"]=-1
                elif kind=="duplicate": bad.loc[0,"ts"]=bad.loc[1,"ts"]
                elif kind=="naive": bad["ts"]=pd.DatetimeIndex(bad.ts).tz_localize(None)
                elif kind=="missing": bad=bad.iloc[:-1]
                elif kind=="unknown": bad.loc[0,"group"]="OTHER"
                else: bad.loc[0,"forecast_kwh"]=np.inf
                bad_path=write_csv(bad,"bad-"+kind+".csv",root=private,index=False)
                with self.assertRaises(ValueError): c.validate_submission(root,cp,bad_path,"synthetic")
            prices["k"]=1
            saved=write_csv(prices,"wrong-k.csv",root=private,index=False); shutil.copyfile(saved,root/"smp.csv")
            with self.assertRaises(ValueError): c.validate_submission(root,cp,fp,"synthetic")


class TestbedContract(unittest.TestCase):
    def test_fixed_year_plan_and_reject_test_period_training(self):
        config=Path(__file__).resolve().parents[1]/"configs/testbed_last_year.json"
        info=testbed.plan(config)
        self.assertEqual(info["unique_models"],9); self.assertEqual(info["maximum_fit_stages"],18)
        self.assertEqual(info["split"],{"cutoff":"2024-05-31","test_start":"2024-06-01","test_end":"2025-05-31"})
        with tempfile.TemporaryDirectory() as tmp:
            bad=read(config); bad["split"]["cutoff"]="2024-06-01"; path=Path(tmp)/"bad.json"; write(path,bad)
            with self.assertRaises(ValueError): testbed.plan(path)
            bad=read(config); bad["candidates"][0]["composition"]={"file":"local.csv","fitted_through":"2024-06-01T00:00:00+09:00"}; write(path,bad)
            with self.assertRaises(ValueError): testbed.plan(path)

    def test_run_resume_identical_support_report_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",48); work=Path(tmp)/"runs"; path=Path(tmp)/"testbed.json"
            self.assertEqual(testbed.last_target_day(root),pd.Timestamp("2024-02-17",tz="Asia/Seoul"))
            spec={"profile":"synthetic","cache_start":"2024-01-01","cache_end":"2024-02-17",
                  "split":{"cutoff":"2024-02-05","test_start":"2024-02-06","test_end":"2024-02-17"},"seeds":[7,11],
                  "baseline":"zero","candidates":[{"id":"zero","model":"zero"},{"id":"tree","model":"lgbm_mae","recipe":"no_weather"}],
                  "training":{"threads":2,"max_rounds":3,"patience":2,"validation_days":5},"statistics":{"replicates":20,"block_days":3},"minimum_common_hours":24}
            comp=pd.DataFrame({"site_id":["SYN-LEFT","SYN-RIGHT"],"group":["G2","G1"]})
            cp=write_csv(comp,"swapped.csv",root=private_root(Path(tmp)/"composition"),index=False)
            spec["candidates"].append({"id":"tree-swapped","model":"lgbm_mae","recipe":"no_weather",
                                       "composition":{"file":str(cp),"fitted_through":"2024-02-05T00:00:00+09:00"}})
            write(path,spec); folder=testbed.run(root,work,path,"synthetic-check")
            result=read(folder/"summary.json")
            self.assertEqual(result["complete_cohort_sites"],2); self.assertEqual(len(result["results"]),6)
            self.assertEqual(len(list((private_root(work)/"models").iterdir())),2)
            self.assertTrue(all(r["settlement"]["common_hours"]==288 for r in result["results"]))
            mean=np.mean([r["settlement"]["W"] for r in result["results"] if r["candidate"]=="tree"])
            self.assertAlmostEqual(next(r for r in result["comparison"] if r["candidate"]=="tree")["mean_W"],mean)
            self.assertTrue(result["test_interval_frozen"]); self.assertFalse(result["test_year_frozen"]); self.assertFalse(result["automatic_promotion"])
            before=sha(folder/"summary.json")
            with patch("oibc_portable.models.fit_tree",side_effect=AssertionError("Completed models must not refit")):
                testbed.run(root,work,path,"synthetic-check",resume=True)
            self.assertEqual(sha(folder/"summary.json"),before)
            html=testbed.report(work,"synthetic-check").read_text(encoding="utf-8")
            self.assertIn("정산금 S",html); self.assertNotIn("SYN-LEFT",html); self.assertNotIn("forecast_kwh",html)
            with self.assertRaises(FileExistsError): testbed.run(root,work,path,"synthetic-check")
            weights=next((private_root(work)/"models").glob("*/refit.txt"))
            with weights.open("a") as stream: stream.write("tampered")
            with self.assertRaises(ValueError): testbed.run(root,work,path,"synthetic-check",resume=True)

    def test_statistics_reject_unequal_support(self):
        def row(candidate,support):
            return {"candidate":candidate,"seed":1,"daily_keys":["2024-01-01"],"support_hours":[support],"daily_R":[100.],"daily_W":[1.],
                    "settlement":{"common_hours":support,"R":100.,"W":1.,"S":99.},"all":{"E":.1},"annual_complete":{"labels":1,"E":.1},"complete_cohort_sites":2}
        with self.assertRaises(ValueError): compare([row("base",24),row("other",23)],"base",[1],1,20,1)
        rows=[row("base",0),row("other",0)]
        for r in rows: r["settlement"]={"common_hours":0,"R":None,"W":None,"S":None}
        self.assertFalse(any(r["ranking_supported"] for r in compare(rows,"base",[1],1,20,1)))


if __name__=="__main__": unittest.main()
