"""pandas vs DuckDB: 같은 집계를 처리 방식만 바꿔 시간과 최대 메모리를 잰다.
각 방식은 별도 프로세스에서 실행하고, 실제 메모리(RSS)가 2.2GB를 넘으면 실패로 기록한다."""
import json, os, resource, subprocess, sys, time
from common import DL, RES
TASKS = {
 "pr_pandas_all": f"""import pandas as pd
df = pd.read_parquet('{DL}/pull_request.parquet')
r = df[df.state=='closed'].groupby('agent').merged_at.apply(lambda s: s.notna().mean())""",
 "pr_pandas_cols": f"""import pandas as pd
df = pd.read_parquet('{DL}/pull_request.parquet', columns=['agent','state','merged_at'])
r = df[df.state=='closed'].groupby('agent').merged_at.apply(lambda s: s.notna().mean())""",
 "pr_duckdb": f"""import duckdb
r = duckdb.sql("select agent, avg(case when merged_at is not null then 1.0 else 0 end) from read_parquet('{DL}/pull_request.parquet') where state='closed' group by 1").fetchall()""",
 "cd_pandas_all": f"""import pandas as pd
df = pd.read_parquet('{DL}/pr_commit_details.parquet')
r = df.groupby('pr_id')[['additions','deletions']].sum()""",
 "cd_pandas_cols": f"""import pandas as pd
df = pd.read_parquet('{DL}/pr_commit_details.parquet', columns=['pr_id','additions','deletions'])
r = df.groupby('pr_id')[['additions','deletions']].sum()""",
 "cd_duckdb": f"""import duckdb
con = duckdb.connect(); con.sql("set memory_limit='1.5GB'")
r = con.sql("select pr_id, sum(additions), sum(deletions) from read_parquet('{DL}/pr_commit_details.parquet') group by 1").fetchall()""",
}
def run(name, cap_mb=2200, reps=int(os.environ.get("REPS", 3))):
    """별도 프로세스로 실행하며 실제 메모리(RSS)를 0.05초마다 잰다. cap_mb를 넘으면 중단(실패로 기록)."""
    import psutil
    code = "import time\nt=time.time()\n%s\nprint('SEC',time.time()-t)" % TASKS[name]
    secs, peak, err = [], 0, None
    for _ in range(reps):
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pp = psutil.Process(p.pid); killed = False
        while p.poll() is None:
            try: rss = pp.memory_info().rss/2**20
            except psutil.NoSuchProcess: break
            peak = max(peak, rss)
            if rss > cap_mb: p.kill(); killed = True; break
            time.sleep(0.05)
        out, e = p.communicate()
        if killed: err = f"메모리 {cap_mb}MB 초과로 중단"; break
        if p.returncode != 0: err = (e.strip().splitlines() or ["?"])[-1][:200]; break
        secs.append(float(out.split("SEC")[-1]))
    ok = err is None
    return {"task": name, "ok": ok, "sec_median": sorted(secs)[len(secs)//2] if ok else None, "runs": len(secs), "peak_rss_mb": round(peak), "error": err}
if __name__ == "__main__":
    names = sys.argv[1:]
    out = f"{RES}/bench.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    for n in names:
        r = run(n); res[n] = r; print(r, flush=True)
        json.dump(res, open(out, "w"), indent=1)
