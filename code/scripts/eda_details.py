import duckdb, json
from common import *
import matplotlib.ticker
con = duckdb.connect()
r0 = f"read_parquet('{ROOT}/data/features_r0.parquet')"; r1 = f"read_parquet('{ROOT}/data/features_r1.parquet')"
d = con.sql(f"""select a.agent, a.y, b.* from {r0} a join {r1} b using (pr_id)""").df()
st = {}
d_all = d
d = d[d.n_files > 0]  # 파일 변경 기록이 없는 PR(주로 Copilot)은 크기·테스트 비교에서 뺀다
g = d.groupby("agent").agg(med_churn=("additions", lambda s: float((s + d.loc[s.index,"deletions"]).median())),
                            med_files=("n_files","median"), test_rate=("has_test","mean")).loc[AGENTS]
st["by_agent"] = g.reset_index().to_dict("records")
xs = [LABEL[a] for a in AGENTS]; cs = [COLOR[a] for a in AGENTS]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].bar(xs, g.med_churn, color=cs, width=0.6)
for i,v in enumerate(g.med_churn): ax[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
ax[0].set_title("PR 크기 (변경 줄 수 중앙값)")
fig.suptitle("파일 변경 기록이 있는 닫힌 PR 기준", fontsize=10, color=INK2, y=0.995); ax[0].set_ylabel("추가+삭제 줄 수")
ax[1].bar(xs, g.test_rate*100, color=cs, width=0.6)
for i,v in enumerate(g.test_rate*100): ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax[1].set_title("테스트 파일을 함께 수정한 PR 비율"); ax[1].set_ylabel("%"); ax[1].set_ylim(0,100)
for x in ax: x.grid(axis="x", visible=False); x.tick_params(axis="x", labelsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_size_test_by_agent.png"); plt.close(fig)
st["n_no_file_changes"] = int((d_all.n_files==0).sum()); st["merge_rate_no_file_changes"] = float(d_all[d_all.n_files==0].y.mean())
st["no_file_by_agent"] = d_all[d_all.n_files==0].agent.value_counts().to_dict(); st["n_code_changing_closed"] = int(len(d))
d2 = d
t = d2.groupby(["agent","has_test"]).y.agg(["mean","count"]).reset_index()
st["merge_by_test"] = t.to_dict("records")
st["merge_by_test_all"] = d2.groupby("has_test").y.agg(["mean","count"]).reset_index().to_dict("records")
import numpy as np
fig, ax = plt.subplots(figsize=(10, 4.4)); w = 0.38; x = np.arange(len(AGENTS))
for k,(ht,lab,alpha) in enumerate([(0,"테스트 없음",0.45),(1,"테스트 포함",1.0)]):
    v = [t[(t.agent==a)&(t.has_test==ht)]["mean"].iloc[0]*100 for a in AGENTS]
    ax.bar(x + (k-0.5)*w, v, w-0.04, color=cs, alpha=alpha, label=lab, edgecolor=SURF, linewidth=1.5)
    for i,vv in enumerate(v): ax.text(x[i]+(k-0.5)*w, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x, xs); ax.set_ylim(0,100); ax.set_ylabel("머지율 (%)"); ax.grid(axis="x", visible=False)
ax.set_title("테스트 포함 여부별 머지율 (코드를 바꾼 PR만, 연한 막대 = 테스트 없음 / 진한 막대 = 포함)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_merge_by_test.png"); plt.close(fig)
json.dump(st, open(f"{RES}/eda_details.json","w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(st, ensure_ascii=False, default=float)[:2500])
