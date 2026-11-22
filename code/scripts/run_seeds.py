"""난수 시드만 바꿔 5번 반복 학습해서, 결과가 우연인지 확인한다.
사용법: python3 run_seeds.py R0 R1  (결과는 results/models_seeds.json에 누적)"""
import json, os, sys, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, RES
from train_eval import SETS, PARAMS, load
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
path = f"{RES}/models_seeds.json"; allres = json.load(open(path)) if os.path.exists(path) else {}
def run(tr, te, X, seed):
    m = xgb.XGBClassifier(**{**PARAMS, "random_state": seed})
    t = time.time(); m.fit(tr[X], tr.y); ts = time.time() - t
    t = time.time(); p = m.predict_proba(te[X])[:, 1]; tp = time.time() - t
    return p, {"auc": roc_auc_score(te.y, p), "prauc_reject": average_precision_score(1-te.y, 1-p), "train_sec": ts, "pred_sec": tp}
for name in sys.argv[1:]:
    cfg = SETS[name]; df = load(cfg); X = cfg["cat"] + cfg["num"]
    tr, te = df[df.split == "train"], df[df.split == "test"]
    rnd = df.sample(frac=1, random_state=0); rtr, rte = rnd.iloc[:len(tr)], rnd.iloc[len(tr):len(tr)+len(te)]
    res = allres.get(name, {}); res.setdefault("time", {}); res.setdefault("random", {}); res.update({"n_train": len(tr), "n_test": len(te), "reject_rate_time": float(1-te.y.mean()), "reject_rate_random": float(1-rte.y.mean()), "n_features": len(X)})
    pp = f"{ROOT}/data/predseeds_{name}.parquet"
    preds = pd.read_parquet(pp) if os.path.exists(pp) else te[["pr_id", "y"]].copy()
    assert (preds.pr_id.to_numpy() == te.pr_id.to_numpy()).all()
    for s in SEEDS:
        p, r = run(tr, te, X, s); res["time"][str(s)] = r; preds[f"p{s}"] = p
        _, r = run(rtr, rte, X, s); res["random"][str(s)] = r
    preds.to_parquet(pp)
    allres[name] = res
    print(name, "time", {k: round(r["auc"], 4) for k, r in res["time"].items()}, "random", {k: round(r["auc"], 4) for k, r in res["random"].items()}, flush=True)
    json.dump(allres, open(path, "w"), indent=1, default=float)
