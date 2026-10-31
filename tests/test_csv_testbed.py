from http.server import ThreadingHTTPServer
from io import StringIO
from pathlib import Path
import json
import tempfile
import threading
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
from synthetic_data import make_dataset
from oibc_portable.csv_testbed import Truth,Store
from oibc_portable.csv_server import handler
from oibc_portable.display import won
from oibc_portable.research_archive import archive,archived_cell
from oibc_portable.research_archive import bundled_results
from importlib.resources import files
from oibc_portable.io import private_root,read,sha


def forecasts(truth,shift=0.):
    parts=[]
    for group,source in (("G1","A"),("G2","B")):
        use=truth.sites.current_group.eq(source).to_numpy()
        values=np.nansum(truth.y[:,use],axis=1)+shift
        parts.append(pd.DataFrame({"group":group,"ts":truth.times,"forecast_kwh":values}))
    return pd.concat(parts,ignore_index=True)


class CSVScoring(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        root=make_dataset(self.base/"data",3)
        self.truth=Truth(root,"synthetic","2024-01-01","2024-01-03")
    def tearDown(self): self.temp.cleanup()

    def test_group_and_site_csv_score_identically_without_training(self):
        grouped=forecasts(self.truth,.123456)
        a=self.truth.score(grouped)
        parts=[]
        for j,key in enumerate(self.truth.sites.site_id):
            parts.append(pd.DataFrame({"site_id":key,"ts":self.truth.times,"pred_kwh":np.nan_to_num(self.truth.y[:,j],nan=0.)+.123456}))
        b=self.truth.score(pd.concat(parts,ignore_index=True))
        self.assertAlmostEqual(a["settlement"]["W"],b["settlement"]["W"],places=5)
        self.assertAlmostEqual(a["settlement"]["S"],b["settlement"]["S"],places=5)
        self.assertEqual(a["settlement"]["common_hours"],int(self.truth.common.sum()))
        self.assertEqual(a["training_time_check"],"unverified")
        self.assertEqual(self.truth.score(forecasts(self.truth))["settlement"]["W"],0.)

    def test_no_candidate_can_drop_a_scorable_hour(self):
        frame=forecasts(self.truth)
        supported=self.truth.times[self.truth.common]
        partial=frame[frame.ts.isin(supported)]
        score=self.truth.score(partial)
        self.assertFalse(score["full_test_grid"])
        self.assertEqual(score["settlement"],self.truth.score(frame)["settlement"])
        with self.assertRaises(ValueError): self.truth.score(partial.iloc[1:])
        for value in (-1.,np.inf,np.nan):
            bad=frame.copy();bad.loc[1,"forecast_kwh"]=value
            with self.assertRaises(ValueError): self.truth.score(bad)
        bad=frame.copy();bad.loc[0,"ts"]=pd.Timestamp("2025-06-01 01:00+09:00")
        with self.assertRaises(ValueError): self.truth.score(bad)
        bad=frame.copy();bad["ts"]=pd.DatetimeIndex(bad.ts).tz_localize(None)
        with self.assertRaises(ValueError): self.truth.score(bad)
        with self.assertRaises(ValueError): self.truth.score(frame,trained_through="2024-06-01")

    def test_persistence_precision_dedup_and_whole_won_csv(self):
        store=Store(self.truth,self.base/"work")
        first=store.add(forecasts(self.truth,.000123456),name="first")
        second=store.add(forecasts(self.truth,.000234567),name="second")
        self.assertNotEqual(first["settlement"]["W"],round(first["settlement"]["W"]))
        store.add(forecasts(self.truth,.000123456),name="first")
        results=Store(self.truth,self.base/"work").results()
        self.assertEqual(len(results),2);self.assertEqual(results[0]["label"],"first")
        self.assertNotIn(".",results[0]["display"]["W"])
        output=store.export_scores();data=pd.read_csv(output,dtype=str)
        self.assertTrue(data.W_KRW.str.fullmatch(r"-?\d+").all())
        self.assertTrue(output.is_relative_to(private_root(self.base/"work")/"details"))
        target=store.folder/(first["id"]+".json")
        with target.open("a") as stream: stream.write("tamper")
        with self.assertRaises(ValueError): store.results()

    def test_money_rounding_and_historical_units(self):
        self.assertEqual(won("1234.5"),"1,235");self.assertEqual(won("-1234.5"),"-1,235")
        self.assertEqual(won(None),"—");self.assertEqual(won([1.5,-1.5]),"[2, -2]")
        self.assertEqual(archived_cell("53.20","W(백만원)"),"53,200,000")
        self.assertEqual(archived_cell("[1234.5, -1.5]","CI7_W"),"[1,235 ~ -2]")
        self.assertEqual(archived_cell("0.1937","W_change_pct"),"0.1937")
        data=archive()
        self.assertEqual(len(data["experiments"]),26)
        self.assertGreaterEqual(len(data["tables"]),187)
        self.assertEqual(len(data["known_settlement"]),5)
        self.assertNotIn(".",data["known_settlement"][0]["display"]["W"])

    def test_dataset_change_rejects_mixed_scoreboard(self):
        with (self.truth.root/"train/smp.csv").open("a") as stream: stream.write("\n")
        with self.assertRaises(RuntimeError): self.truth.score(forecasts(self.truth))

    def test_bundled_scores_require_original_dataset_fingerprint(self):
        data=json.loads(files("oibc_portable").joinpath("historical_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(len(bundled_results(data["source_context"])),170)
        self.assertEqual(bundled_results(self.truth.context),[])
        wrong=dict(data["source_context"],version="other-rules")
        self.assertEqual(bundled_results(wrong),[])
        self.assertTrue(all(r["settlement"]["common_hours"]==3802 for r in data["results"]))

    def test_html_whole_won_keeps_raw_json_precision(self):
        from oibc_portable.testbed_report import render
        row={"candidate":"fixture","mean_S":1234.5,"mean_W":3456.5,"mean_R":4691.,"delta_S_vs_baseline":1.5,
             "delta_S_exploratory_95pct":[-1.5,2.5],"common_hours":48,"mean_E":.123456,"mean_annual_complete_E":None,
             "status":"reference","seed_wins":0,"seeds":1}
        result={"comparison":[row],"results":[],"profile":"synthetic","split":{"test_start":"2024-01-01","test_end":"2024-01-03","cutoff":"2023-12-31"},"complete_cohort_sites":2,"statistics":{"block_days":3,"replicates":20}}
        html=render(self.base,result).read_text(encoding="utf-8")
        self.assertIn("1,235",html);self.assertIn("3,457",html);self.assertNotIn("1,234.5000",html)
        self.assertEqual(read(self.base/"summary.json")["comparison"][0]["mean_S"],1234.5)

    def test_local_http_upload_and_origin_protection(self):
        store=Store(self.truth,self.base/"http")
        server=ThreadingHTTPServer(("127.0.0.1",0),handler(store,"test-token",0))
        server.RequestHandlerClass=handler(store,"test-token",server.server_port)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f"http://127.0.0.1:{server.server_port}"
        try:
            page=urlopen(url).read().decode()
            self.assertIn("CSV 채점",page);self.assertIn("test-token",page)
            context=json.load(urlopen(url+"/api/context"));self.assertEqual(context["hours"],72)
            payload=json.dumps({"forecast_csv":forecasts(self.truth,.1234).to_csv(index=False),"name":"HTTP fixture"}).encode()
            request=Request(url+"/api/score",payload,{"Content-Type":"application/json","X-OIBC-Token":"test-token","Origin":url})
            self.assertTrue(json.load(urlopen(request))["ok"])
            rows=json.load(urlopen(url+"/api/results"));self.assertEqual(len(rows),1)
            self.assertNotIn("SYN-LEFT",json.dumps(rows))
            for headers in ({"X-OIBC-Token":"wrong"},{"X-OIBC-Token":"test-token","Origin":"https://invalid.example"}):
                with self.assertRaises(HTTPError) as caught: urlopen(Request(url+"/api/score",payload,headers))
                self.assertEqual(caught.exception.code,403)
            exported=urlopen(url+"/api/export").read().decode();self.assertIn("W_KRW",exported)
            self.assertEqual(len(json.load(urlopen(url+"/api/history"))["experiments"]),26)
        finally: server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=="__main__": unittest.main()
