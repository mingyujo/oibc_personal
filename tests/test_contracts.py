import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from oibc_portable import features as f
from oibc_portable import models
from oibc_portable.io import RunLock, private_root
from oibc_portable.local_details import write_csv
from oibc_portable.evaluation import settlement
from synthetic_data import make_dataset


class Contracts(unittest.TestCase):
    def test_complete_feature_registry_and_true_no_weather_control(self):
        self.assertEqual(len(f.names()), 262); self.assertEqual(len(f.BASE), 137)
        selected = f.recipe_indices("no_weather")
        self.assertEqual(len(selected), 15)
        self.assertEqual(set(selected), set(range(9)) | set(range(208,214)))

    def test_inner_origin_embargo_and_midnight(self):
        days = pd.date_range("2024-01-01", "2024-02-28", tz=f.TZ)
        train, valid = models.masks(days, "2024-02-29", 30)
        first = days[valid].min(); origin = first-pd.Timedelta("1d")+pd.Timedelta("17h")
        self.assertTrue((days[train]+pd.Timedelta("1d") <= origin).all())
        self.assertEqual(int(valid.sum()), 30); self.assertEqual(int((~train & ~valid).sum()), 1)

    def test_midnight_label_belongs_to_preceding_target_day(self):
        t = f.timestamps(["2024-03-01 00:00:00+09:00"])
        self.assertEqual((t-pd.Timedelta("1ns")).normalize()[0], f.stamp("2024-02-29"))

    def test_neighbor_never_crosses_target_day(self):
        x = np.arange(48, dtype="float32").reshape(2,24,1)
        shifted = f.shift(x, 1)
        self.assertTrue(np.isnan(shifted[:,23]).all()); self.assertEqual(shifted[0,22,0], x[0,23,0])

    def test_clipped_metric_matches_final_output_and_weights(self):
        p = np.array([-2., 3.]); y = np.array([0., 1.]); w = np.array([100., 200.])
        self.assertAlmostEqual(models.clipped_mae(p,y,w), 400/300)
        metric = models.ClippedMAE(); total, exposure = metric.evaluate([p], y, w)
        self.assertAlmostEqual(metric.get_final_error(total, exposure), 400/300)

    def test_source_data_not_mutated_missing_labels_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp)/"dataset", 4)
            from oibc_portable.io import sha
            before = sha(root/"train/generation.parquet")
            x,y,days,_ = f.build(root,"2024-01-01","2024-01-04")
            self.assertEqual(x.shape, (8,24,262)); self.assertEqual(int((~np.isfinite(y)).sum()),3)
            self.assertEqual(sha(root/"train/generation.parquet"),before)

    def test_prediction_features_do_not_read_generation_or_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_dataset(Path(tmp)/"dataset",4)
            real = pd.read_parquet
            def guarded(path,*a,**kw):
                self.assertNotIn("generation",str(path)); return real(path,*a,**kw)
            with patch.object(pd,"read_parquet",guarded): x,y,_,_ = f.build(root,"2024-01-01","2024-01-01",split="test",labels=False)
            self.assertTrue(np.isnan(y).all()); self.assertEqual(x.shape[-1],262)

    def test_missing_issue_is_distinct_from_variable_missing_and_uses_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",2)
            import pandas as pd
            p=root/"train/forecast_wa.parquet"; frame=pd.read_parquet(p)
            chosen=(frame.wa_station.eq("SYN-A1") & frame["round"].eq(17) & frame.ts.eq(f.stamp("2024-01-01 11:00")))
            frame.loc[~chosen].to_parquet(p,index=False)
            x,_,_,_=f.build(root,"2024-01-01","2024-01-02")
            # The first site in sorted order is SYN-LEFT.
            self.assertEqual(x[0,10,f.BASE.index("wa_r17_missing")],1)
            self.assertEqual(x[0,10,f.BASE.index("wa_humidity_r17_is_missing")],1)
            self.assertEqual(x[0,10,214+1],x[0,10,f.BASE.index("wa_humidity_r10")])

    def test_off_hour_forecast_is_excluded_without_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",2); path=root/"train/forecast_wb.parquet"
            frame=pd.read_parquet(path); extra=frame.iloc[[0]].copy(); extra["ts"]+=pd.Timedelta("15min")
            pd.concat([frame,extra],ignore_index=True).to_parquet(path,index=False)
            _,_,_,audit=f.build(root,"2024-01-01","2024-01-02")
            self.assertEqual(audit["off_hour_forecasts_excluded"]["wb"],1)

    def test_csv_path_cannot_escape_private_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = private_root(tmp)
            with self.assertRaises(ValueError): write_csv(pd.DataFrame({"a":[1]}), "../escape.csv", root=root,index=False)

    def test_live_job_cannot_be_resumed_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLock(tmp):
                with self.assertRaises(RuntimeError):
                    with RunLock(tmp,resume=True): pass

    def test_no_common_settlement_labels_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",2); sites=f.load_sites(root)
            _,y,days,_=f.build(root,"2024-01-01","2024-01-01")
            y[:]=np.nan; prices=pd.Series(100.,index=pd.date_range("2024-01-01 01:00",periods=24,freq="h",tz=f.TZ))
            result=settlement(np.ones_like(y),y,days,sites,prices)
            self.assertIsNone(result["W"]); self.assertEqual(result["common_hours"],0)


if __name__ == "__main__": unittest.main()
