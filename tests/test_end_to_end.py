import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from oibc_portable import data, models, evaluation
from oibc_portable.io import private_root, read, sha
from synthetic_data import make_dataset


class EndToEnd(unittest.TestCase):
    def test_cpu_tree_training_resume_evaluation_and_guarded_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",48,official=True); work=private_root(Path(tmp)/"runs")
            data.prepare(root,work,"2024-01-01","2024-02-17")
            data.prepare(root,work,"2024-01-01","2024-02-17",resume=True)
            for name,recipe in (("lgbm_mae","raw6"),("lgbm_mae","no_weather"),("catboost_mae","raw6"),("catboost_rmse","raw6")):
                cfg=models.Config(model=name,recipe=recipe,cutoff="2024-02-05",threads=2,max_rounds=5,patience=2,validation_days=5)
                folder=models.train(work,cfg)
                self.assertEqual(read(folder/"time-audit.json")["inner_origin_margin_hours"],17)
                before=sha(models.artifact(folder,cfg)); models.train(work,cfg,resume=True)
                self.assertEqual(sha(models.artifact(folder,cfg)),before)
                score=evaluation.evaluate(work,folder,"2024-02-06","2024-02-17")
                self.assertGreater(read(score)["all"]["labels"],0)
                with self.assertRaises(ValueError): evaluation.evaluate(work,folder,"2024-02-01","2024-02-17")
                with self.assertRaises(FileExistsError): models.train(work,cfg)
                if name=="lgbm_mae" and recipe=="no_weather":
                    out=evaluation.predict_submission(root,work,folder)
                    self.assertEqual(read(out/"verification.json")["forecast_rows"],17520)
                    self.assertTrue(out.is_relative_to(work/"details"))
                    with models.artifact(folder,cfg).open("a") as stream: stream.write("tampered")
                    with self.assertRaises(ValueError): models.load(folder)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Optional neural dependency not installed")
    def test_optional_neural_models_train_reload_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_dataset(Path(tmp)/"dataset",12); work=private_root(Path(tmp)/"runs")
            data.prepare(root,work,"2024-01-01","2024-01-12")
            for name in ("mlp24","patch3"):
                cfg=models.Config(model=name,recipe="raw6",cutoff="2024-01-10",threads=2,validation_days=2,epochs=1,neural_patience=1,batch_size=4)
                folder=models.train(work,cfg); net,config,_=models.load(folder)
                x=next(data.chunks(work,"2024-01-11","2024-01-12"))[0]
                pred=models.predict(net,config,x,folder)
                self.assertEqual(pred.shape,(4,24)); self.assertTrue(np.isfinite(pred).all()); self.assertTrue((pred>=0).all())


if __name__ == "__main__": unittest.main()
