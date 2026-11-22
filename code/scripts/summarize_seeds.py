"""5개 시드 결과를 평균·범위로 요약하고, 평가 PR을 다시 뽑는 부트스트랩으로 차이의 95% 신뢰구간을 구한다."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import ROOT, RES
S = json.load(open(f"{RES}/models_seeds.json")); Q = json.load(open(f"{RES}/r2_seeds.json"))
def summ(runs, key):
    v = np.array([r[key] for r in runs.values()]); return {"mean": v.mean(), "min": v.min(), "max": v.max(), "sd": v.std(ddof=1)}
out = {"full": {}, "sample": {}}
for n, r in S.items():
    out["full"][n] = {"time_auc": summ(r["time"], "auc"), "random_auc": summ(r["random"], "auc"), "time_prauc": summ(r["time"], "prauc_reject"),
                      "train_sec": summ(r["time"], "train_sec"), "n_train": r["n_train"], "n_test": r["n_test"], "reject_rate_time": r["reject_rate_time"]}
    out["full"][n]["gap_random_minus_time"] = out["full"][n]["random_auc"]["mean"] - out["full"][n]["time_auc"]["mean"]
for n in ["R0", "R1", "R2"]:
    runs = {k: v for k, v in Q[n].items() if k != "n_features"}
    out["sample"][n] = {"auc": summ(runs, "auc"), "prauc": summ(runs, "prauc_reject"), "train_sec": summ(runs, "train_sec"), "pred_sec": summ(runs, "pred_sec"), "n_features": Q[n]["n_features"]}
def boot(y, pa, pb, B=1000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
def paired(a, b, cols_a, cols_b, y):
    per_seed = [roc_auc_score(y, b[cb]) - roc_auc_score(y, a[ca]) for ca, cb in zip(cols_a, cols_b)]
    pa = a[cols_a].mean(axis=1).to_numpy(); pb = b[cols_b].mean(axis=1).to_numpy()
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "boot95_seedavg": boot(y, pa, pb, B=500)}
ps = pd.read_parquet(f"{ROOT}/data/predseeds_sample.parquet"); y = ps.y.to_numpy()
sd = range(5)
out["sample"]["R1_minus_R0"] = paired(ps, ps, [f"R0_p{i}" for i in sd], [f"R1_p{i}" for i in sd], y)
out["sample"]["R2_minus_R1"] = paired(ps, ps, [f"R1_p{i}" for i in sd], [f"R2_p{i}" for i in sd], y)
for a, b in [("R0", "R1"), ("R0_single", "R1_single")]:
    A = pd.read_parquet(f"{ROOT}/data/predseeds_{a}.parquet"); Bp = pd.read_parquet(f"{ROOT}/data/predseeds_{b}.parquet")
    assert (A.pr_id.to_numpy() == Bp.pr_id.to_numpy()).all()
    out["full"][f"{b}_minus_{a}"] = paired(A, Bp, [f"p{i}" for i in sd], [f"p{i}" for i in sd], A.y.to_numpy())
json.dump(out, open(f"{RES}/summary_seeds.json", "w"), indent=1, default=float)
def r(x): return round(x, 4)
for k in ["R0", "R1", "R0_single", "R1_single"]:
    f = out["full"][k]; print(k, "time", r(f["time_auc"]["mean"]), (r(f["time_auc"]["min"]), r(f["time_auc"]["max"])), "rand", r(f["random_auc"]["mean"]), "gap", r(f["gap_random_minus_time"]), "prauc", r(f["time_prauc"]["mean"]), "train", r(f["train_sec"]["mean"]))
for k in ["R0", "R1", "R2"]:
    f = out["sample"][k]; print("S", k, r(f["auc"]["mean"]), (r(f["auc"]["min"]), r(f["auc"]["max"])), "prauc", r(f["prauc"]["mean"]), (r(f["prauc"]["min"]), r(f["prauc"]["max"])), "train", r(f["train_sec"]["mean"]), "pred", r(f["pred_sec"]["mean"]))
for k in ["R1_minus_R0", "R2_minus_R1"]: print("S", k, out["sample"][k])
for k in ["R1_minus_R0", "R1_single_minus_R0_single"]: print("F", k, out["full"][k])
