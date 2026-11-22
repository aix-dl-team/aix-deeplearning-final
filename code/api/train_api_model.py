"""API에서 쓸 R1 모델을 만든다.
성능 검증은 시간 분할 실험(과거로 학습, 미래로 평가)으로 이미 했으므로(AUC 약 0.727),
실제로 쓸 모델은 정답이 있는 닫힌 PR 전부(2024-12 ~ 2026-03)로 다시 학습한다. 설정은 실험과 같다."""
import json, sys, time, numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, "../scripts")
from common import ROOT
from train_eval import PARAMS
from pr_features import R1_CAT, R1_NUM
VERSION = "r1-xgb-20260331"
df = pd.read_parquet(f"{ROOT}/data/features_r0.parquet").merge(pd.read_parquet(f"{ROOT}/data/features_r1.parquet"), on="pr_id")
df = df.sort_values("pr_id").reset_index(drop=True)
cats = {}
for c in R1_CAT:
    top = [str(x) for x in df[c].value_counts().index[:30]]
    cats[c] = top + ["other_"]
    df[c] = pd.Categorical(np.where(df[c].isin(top), df[c], "other_"), categories=cats[c])
X = R1_CAT + R1_NUM
m = xgb.XGBClassifier(**{**PARAMS, "random_state": 0}); t = time.time(); m.fit(df[X], df.y)
m.save_model("model_r1.json")
meta = {"model_version": VERSION, "features": X, "categorical": cats, "n_train": int(len(df)),
        "train_period": [str(df.created_at.min())[:10], str(df.created_at.max())[:10]], "train_merge_rate": float(df.y.mean()),
        "params": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
        "expected_auc_time_split": 0.727, "note": "성능 숫자는 과거 PR로 학습하고 미래 PR로 평가한 실험(5번 평균) 기준"}
json.dump(meta, open("model_meta.json", "w"), ensure_ascii=False, indent=1)
print(VERSION, len(df), round(time.time() - t, 1), "s")
