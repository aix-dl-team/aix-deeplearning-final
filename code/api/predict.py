"""R1 모델로 PR 하나의 머지 확률을 계산한다. 백엔드(API)는 이 클래스를 불러 쓰면 된다.

사용 예:
    from predict import MergePredictor
    model = MergePredictor()                  # 서버 시작할 때 한 번만
    result = model.predict(pr_json)           # 요청마다
    # {"merge_probability": 0.81, "model_version": "r1-xgb-20260331", "elapsed_ms": 3.2}
"""
import json, os, time
import numpy as np, pandas as pd, xgboost as xgb
from pr_features import build_features

HERE = os.path.dirname(os.path.abspath(__file__))


class MergePredictor:
    def __init__(self, model_dir=HERE):
        self.meta = json.load(open(os.path.join(model_dir, "model_meta.json"), encoding="utf-8"))
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "model_r1.json"))
        self.features = self.meta["features"]
        self.cats = self.meta["categorical"]

    def _frame(self, feats):
        df = pd.DataFrame([feats])
        for c, allowed in self.cats.items():  # 학습 때 상위 30개에 없던 값은 "other_"로 바꾼다
            v = df[c].astype(object).where(df[c].isin(allowed), "other_")
            df[c] = pd.Categorical(v, categories=allowed)
        for c in self.features:
            if c not in self.cats:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)  # None → NaN(빈 값)
        return df[self.features]

    def predict_features(self, feats):
        return float(self.model.predict_proba(self._frame(feats))[:, 1][0])

    def predict(self, pr):
        t = time.perf_counter()
        p = self.predict_features(build_features(pr))
        return {"merge_probability": round(p, 4), "model_version": self.meta["model_version"],
                "elapsed_ms": round((time.perf_counter() - t) * 1000, 1)}


if __name__ == "__main__":
    import sys
    pr = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else json.load(open(os.path.join(HERE, "example_request.json"), encoding="utf-8"))
    print(json.dumps(MergePredictor().predict(pr), ensure_ascii=False))
