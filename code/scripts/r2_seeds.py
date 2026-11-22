"""같은 표본(학습 3만, 평가 3만)에서 R0, R1, R2를 난수 시드 5개로 반복해 비교한다 (r2_sample.py와 같은 설정).
TF-IDF+SVD 결과는 data/r2_svd_cache.parquet에 저장해 다시 쓴다."""
import json, time, os, glob
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, RES
from train_eval import PARAMS
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
f = pd.read_parquet(f"{ROOT}/data/features_r0.parquet").merge(pd.read_parquet(f"{ROOT}/data/features_r1.parquet"), on="pr_id")
s = f.merge(pd.read_parquet(f"{ROOT}/data/sample_ids.parquet")[["pr_id"]], on="pr_id").sort_values("pr_id").reset_index(drop=True)
zc = [f"svd{i}" for i in range(128)]
cache = f"{ROOT}/data/r2_svd_cache.parquet"; path = f"{RES}/r2_seeds.json"
out = json.load(open(path)) if os.path.exists(path) else {}
if os.path.exists(cache):
    z = pd.read_parquet(cache); s = s.merge(z, on="pr_id")
else:
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/data/diff/chunk_*.parquet"))]
    T = {"extract_sec": float(sum(p["extract_sec"].iloc[0] for p in parts))}
    s = s.merge(pd.concat(parts)[["pr_id", "diff"]], on="pr_id", how="left"); s["diff"] = s["diff"].fillna("")
    tr0 = s.split == "train"
    t = time.time()
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]{1,30}", lowercase=True, min_df=5, max_features=50000, sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(s.loc[tr0, "diff"]); Xte = vec.transform(s.loc[~tr0, "diff"]); T["tfidf_sec"] = time.time() - t
    t = time.time(); svd = TruncatedSVD(128, random_state=0); Ztr = svd.fit_transform(Xtr); Zte = svd.transform(Xte); T["svd_sec"] = time.time() - t
    Z = np.zeros((len(s), 128), dtype=np.float32); Z[tr0.to_numpy()] = Ztr; Z[~tr0.to_numpy()] = Zte
    s[zc] = Z; s = s.drop(columns="diff"); s[["pr_id"] + zc].to_parquet(cache)
    out.update({"times": T, "vocab": len(vec.vocabulary_), "empty_diff_share": None})
tr, te = s[s.split == "train"].copy(), s[s.split == "test"].copy()
for c in ["agent", "lang", "task", "top_ext"]:
    top = tr[c].value_counts().index[:30]
    for d in (tr, te): d[c] = pd.Categorical(np.where(d[c].isin(top), d[c], "other_"), categories=list(top) + ["other_"])
R0 = ["agent", "lang", "task", "stars", "forks", "repo_prior_prs", "title_len", "body_len"]
R1 = R0 + ["top_ext", "n_files", "additions", "deletions", "log_churn", "n_ext", "has_test", "test_line_share", "doc_file_share"]
out.update({"n_train": len(tr), "n_test": len(te), "reject_rate_test": float(1 - te.y.mean())})
pp = f"{ROOT}/data/predseeds_sample.parquet"
preds = pd.read_parquet(pp) if os.path.exists(pp) else te[["pr_id", "y"]].copy()
for name, X in [("R0", R0), ("R1", R1), ("R2", R1 + zc)]:
    out.setdefault(name, {"n_features": len(X)})
    for sd in SEEDS:
        m = xgb.XGBClassifier(**{**PARAMS, "random_state": sd})
        t = time.time(); m.fit(tr[X], tr.y); ts = time.time() - t
        t = time.time(); p = m.predict_proba(te[X])[:, 1]; tp = time.time() - t
        out[name][str(sd)] = {"auc": roc_auc_score(te.y, p), "prauc_reject": average_precision_score(1 - te.y, 1 - p), "train_sec": ts, "pred_sec": tp}
        preds[f"{name}_p{sd}"] = p
        print(name, sd, round(out[name][str(sd)]["auc"], 4), round(ts, 1), flush=True)
    json.dump(out, open(path, "w"), indent=1, default=float); preds.to_parquet(pp)
