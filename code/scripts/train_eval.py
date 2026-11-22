"""같은 분류기(XGBoost, 고정 설정)로 피처 세트만 바꿔 학습·평가하고 시간과 성능을 기록한다.
사용법: python3 train_eval.py R0 [R1 ...]"""
import json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, RES
SETS = {
 "R0": {"files": ["features_r0"], "cat": ["agent","lang","task"],
        "num": ["stars","forks","repo_prior_prs","title_len","body_len"]},
}
extra = os.path.join(ROOT, "scripts", "sets_extra.json")
if os.path.exists(extra): SETS.update(json.load(open(extra)))
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
              tree_method="hist", enable_categorical=True, n_jobs=4, random_state=42, eval_metric="auc")
def load(cfg):
    df = None
    for f in cfg["files"]:
        d = pd.read_parquet(f"{ROOT}/data/{f}.parquet")
        df = d if df is None else df.merge(d, on="pr_id", how="inner", suffixes=("", "_dup"))
    if "filter" in cfg: df = df.query(cfg["filter"])
    df = df.sort_values("pr_id").reset_index(drop=True)  # 파일 행 순서와 무관하게 재현되도록 정렬
    for c in cfg["cat"]:
        top = df[c].value_counts().index[:30]
        df[c] = pd.Categorical(np.where(df[c].isin(top), df[c], "other_"))
    return df
def fit_eval(tr, te, X):
    m = xgb.XGBClassifier(**PARAMS)
    t = time.time(); m.fit(tr[X], tr.y); t_train = time.time() - t
    t = time.time(); p = m.predict_proba(te[X])[:, 1]; t_pred = time.time() - t
    return m, p, {"auc": roc_auc_score(te.y, p), "prauc_reject": average_precision_score(1 - te.y, 1 - p),
                  "reject_rate": float(1 - te.y.mean()), "train_sec": t_train,
                  "pred_ms_per_1k": t_pred / len(te) * 1e6, "n_train": len(tr), "n_test": len(te)}
if __name__ == "__main__":
    res_path = f"{RES}/models.json"
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for name in sys.argv[1:]:
        cfg = SETS[name]; df = load(cfg); X = cfg["cat"] + cfg["num"]
        tr, te = df[df.split == "train"], df[df.split == "test"]
        m, p, r_time = fit_eval(tr, te, X)
        # 무작위 분할: 학습·평가 개수를 시간 분할과 똑같이 맞춰 데이터 양 차이를 없앤다
        rnd = df.sample(frac=1, random_state=0)
        _, _, r_rand = fit_eval(rnd.iloc[:len(tr)], rnd.iloc[len(tr):len(tr)+len(te)], X)
        g = m.get_booster().get_score(importance_type="gain"); imp = sorted(((x, g.get(x, 0)) for x in X), key=lambda t: -t[1])[:10]
        allres[name] = {"time_split": r_time, "random_split": r_rand, "top_gain": imp, "n_features": len(X)}
        test_pred = te[["pr_id","agent","y"]].assign(p=p); test_pred.to_parquet(f"{ROOT}/data/pred_{name}.parquet")
        print(name, json.dumps(allres[name], default=float)[:900], flush=True)
        json.dump(allres, open(res_path, "w"), indent=1, default=float)
