"""표본 PR의 diff 텍스트를 1만 개씩 조각으로 추출해 저장한다. 이미 만든 조각은 건너뛴다 (이어서 실행 가능)."""
import os, time, json, pandas as pd, duckdb
from common import ROOT, DL
CAP, STEP, BUDGET = 6000, 10000, 140
ids = pd.read_parquet(f"{ROOT}/data/sample_ids.parquet")["pr_id"].to_numpy()
os.makedirs(f"{ROOT}/data/diff", exist_ok=True)
con = duckdb.connect(); con.sql("set memory_limit='1.5GB'"); con.sql("set threads=2"); con.sql("set preserve_insertion_order=false")
con.sql("set enable_progress_bar=false"); con.sql("set temp_directory='/tmp/duck'")
t0 = time.time()
for k in range(0, len(ids), STEP):
    out = f"{ROOT}/data/diff/chunk_{k:06d}.parquet"
    if os.path.exists(out): continue
    if time.time() - t0 > BUDGET - 60: print("budget stop"); break
    t = time.time()
    con.register("ids", pd.DataFrame({"pr_id": ids[k:k+STEP]}))
    df = con.sql(f"""with r as (select d.pr_id, 'FILE ' || d.filename || ' ' || left(coalesce(d.patch,''), 1500) as chunk
        from read_parquet('{DL}/pr_commit_details.parquet') d semi join ids using (pr_id))
      select pr_id, left(string_agg(chunk, ' '), {CAP}) as diff from r group by pr_id""").df()
    df["extract_sec"] = time.time() - t
    df.to_parquet(out); print("chunk", k, round(time.time()-t, 1), flush=True)
print("done", sorted(os.listdir(f"{ROOT}/data/diff")))
