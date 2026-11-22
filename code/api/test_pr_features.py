"""pr_features.py가 학습 피처 파일(features_r0/r1.parquet)과 같은 값을 내는지, 무작위 PR 3,000개로 확인한다."""
import sys, math, json, duckdb, numpy as np, pandas as pd
sys.path.insert(0, "../scripts")
from common import ROOT, pq
import pr_features as P
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
f0 = pd.read_parquet(f"{ROOT}/data/features_r0.parquet"); f1 = pd.read_parquet(f"{ROOT}/data/features_r1.parquet")
f = f0.merge(f1, on="pr_id")
# 일부러 까다로운 PR(커밋 여러 개, 파일 이름 빈 기록, 테스트 포함)을 섞어 뽑는다
pick = pd.concat([f.sample(N // 2, random_state=1), f[f.n_commits > 1].sample(N // 4, random_state=2),
                  f[(f.n_null_files > 0) | (f.has_test == 1)].sample(N // 4, random_state=3)]).drop_duplicates("pr_id")
con = duckdb.connect(); con.sql("set memory_limit='1.5GB'"); con.register("ids", pick[["pr_id"]])
rows = con.sql(f"select pr_id, sha, filename, additions, deletions from {pq('pr_commit_details')} d semi join ids using (pr_id)").df()
prs = con.sql(f"select p.id pr_id, p.agent, p.title, p.body, r.stars, r.forks, r.language from {pq('pull_request')} p left join {pq('repository')} r on r.id=p.repo_id semi join ids on ids.pr_id=p.id").df()
prs = prs.set_index("pr_id"); g = {k: v for k, v in rows.groupby("pr_id")}
bad = {}; ok = 0
for _, t in pick.iterrows():
    pr0 = prs.loc[t.pr_id]; rr = g[t.pr_id]
    commits = [{"sha": s, "files": [{"filename": (None if pd.isna(x.filename) else x.filename),
                                      "additions": (None if pd.isna(x.additions) else int(x.additions)),
                                      "deletions": (None if pd.isna(x.deletions) else int(x.deletions))} for x in c.itertuples()]}
               for s, c in rr.groupby("sha", sort=False)]
    pr = {"agent": pr0.agent, "title": None if pd.isna(pr0.title) else pr0.title, "body": None if pd.isna(pr0.body) else pr0.body,
          "repo": {"stars": None if pd.isna(pr0.stars) else int(pr0.stars), "forks": None if pd.isna(pr0.forks) else int(pr0.forks),
                   "language": None if pd.isna(pr0.language) else pr0.language},
          "repo_prior_prs": int(t.repo_prior_prs), "commits": commits}
    got = P.build_features(pr); diff = []
    for k, v in got.items():
        w = t[k]
        wn = w is None or (isinstance(w, float) and np.isnan(w))
        if v is None or wn: same = (v is None and wn)
        elif isinstance(v, str) or isinstance(w, str): same = (v == w)
        else: same = math.isclose(float(v), float(w), rel_tol=1e-9, abs_tol=1e-9)
        if not same: diff.append(k)
    if diff:
        for k in diff: bad.setdefault(k, []).append((int(t.pr_id), got[k], t[k]))
    else: ok += 1
print(json.dumps({"n": len(pick), "all_equal": ok, "mismatch_by_feature": {k: len(v) for k, v in bad.items()},
                  "examples": {k: [str(x) for x in v[:3]] for k, v in bad.items()}}, ensure_ascii=False, indent=1))
