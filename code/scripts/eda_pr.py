import duckdb, json
from common import *
import matplotlib.ticker
con = duckdb.connect()
con.sql(f"create view pr as select * from {pq('pull_request')}")
con.sql(f"create view repo as select * from {pq('repository')}")
con.sql(f"create view cm as select * from {pq('pr_commits')}")
stats = {}
a = con.sql("""select agent, count(*) n, count(merged_at) merged,
  sum(case when state='closed' then 1 else 0 end) closed from pr group by 1""").df().set_index("agent").loc[AGENTS]
a["merge_rate"] = a.merged / a.closed
stats["by_agent"] = a.reset_index().to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
xs = [LABEL[g] for g in AGENTS]; cs = [COLOR[g] for g in AGENTS]
ax[0].bar(xs, a.n, color=cs, width=0.6)
for i,v in enumerate(a.n): ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK2)
ax[0].set_title("에이전트별 PR 수 (AIDev-pop)"); ax[0].set_ylabel("PR 수")
ax[1].bar(xs, a.merge_rate*100, color=cs, width=0.6)
for i,v in enumerate(a.merge_rate*100): ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax[1].set_title("머지율 (닫힌 PR 중 머지된 비율)"); ax[1].set_ylabel("%"); ax[1].set_ylim(0, 100)
for x in ax:
    x.grid(axis="x", visible=False); x.tick_params(axis="x", labelsize=10)
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_agent_count_merge.png"); plt.close(fig)
m = con.sql("select substr(created_at,1,7) mon, agent, count(*) n from pr group by 1,2 order by 1").df()
p = m.pivot(index="mon", columns="agent", values="n").fillna(0)[AGENTS]
fig, ax = plt.subplots(figsize=(11, 4.4))
for g in AGENTS: ax.plot(p.index, p[g], color=COLOR[g], lw=2, label=LABEL[g], marker="o", ms=4)
ax.set_title("월별 PR 수 추이 (에이전트별)"); ax.set_ylabel("PR 수"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:,.0f}")); ax.legend(frameon=False, ncol=3)
ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(f"{FIG}/fig2_monthly_volume.png"); plt.close(fig)
stats["monthly_total"] = con.sql("select substr(created_at,1,7) mon, count(*) n from pr group by 1 order by 1").fetchall()
c = con.sql("""with k as (select pr_id, count(*) k from cm group by 1)
  select p.agent, count(*) n, avg(case when k.k=1 then 1.0 else 0 end) single, median(k.k) med
  from pr p join k on k.pr_id=p.id group by 1""").df().set_index("agent").loc[AGENTS]
stats["commits"] = c.reset_index().to_dict("records")
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.bar(xs, c.single*100, color=cs, width=0.6)
for i,v in enumerate(c.single*100): ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_title("커밋이 1개뿐인 PR 비율"); ax.set_ylabel("%"); ax.set_ylim(0,100); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_single_commit.png"); plt.close(fig)
s = con.sql("""select case when r.stars<500 then '101–499' when r.stars<2000 then '500–1,999'
  when r.stars<10000 then '2,000–9,999' else '10,000+' end b,
  case when r.stars<500 then 1 when r.stars<2000 then 2 when r.stars<10000 then 3 else 4 end o,
  sum(case when p.state='closed' then 1 else 0 end) n, count(p.merged_at)*1.0/sum(case when p.state='closed' then 1 else 0 end) mr
  from pr p join repo r on r.id=p.repo_id group by 1,2 order by o""").df()
stats["by_stars"] = s.to_dict("records")
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(s.b, s.mr*100, color="#2a78d6", width=0.6)
for i,(v,n) in enumerate(zip(s.mr*100, s.n)): ax.text(i, v, f"{v:.1f}%\n(닫힌 PR {int(n):,}개)", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_title("저장소 스타 수 구간별 머지율 (닫힌 PR 기준)"); ax.set_xlabel("저장소 스타 수"); ax.set_ylabel("%"); ax.set_ylim(0,100); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_merge_by_stars.png"); plt.close(fig)
json.dump(stats, open(f"{RES}/eda_pr.json","w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(stats, ensure_ascii=False, default=str)[:3000])
