"""R0: 코드를 보지 않는 피처 (PR이 열릴 때 알 수 있는 메타데이터)."""
import duckdb, time
from common import *
t = time.time()
con = duckdb.connect()
con.sql(f"""copy (
 with pr as (select * from {pq('pull_request')}),
 hist as (select id, count(*) over (partition by repo_id order by created_at
            range between unbounded preceding and current row) - 1 as repo_prior_prs from pr)
 select p.id as pr_id, p.agent, p.created_at,
   case when p.created_at < '2026-01-01' then 'train' when p.created_at < '2026-02-01' then 'valid' else 'test' end as split,
   cast(p.merged_at is not null as integer) as y,
   coalesce(r.stars,0) as stars, coalesce(r.forks,0) as forks, coalesce(r.language,'unknown') as lang,
   h.repo_prior_prs,
   length(coalesce(p.title,'')) as title_len, length(coalesce(p.body,'')) as body_len,
   case when regexp_matches(lower(p.title), '^(feat|feature)') then 'feat'
        when regexp_matches(lower(p.title), '^(fix|bug|hotfix)') then 'fix'
        when regexp_matches(lower(p.title), '^docs?') then 'docs'
        when regexp_matches(lower(p.title), '^tests?') then 'test'
        when regexp_matches(lower(p.title), '^refactor') then 'refactor'
        when regexp_matches(lower(p.title), '^(chore|build|ci|deps)') then 'chore'
        when regexp_matches(lower(p.title), '^perf') then 'perf'
        else 'other' end as task
 from pr p left join {pq('repository')} r on r.id = p.repo_id
 join hist h on h.id = p.id
 where p.state = 'closed'
) to '{ROOT}/data/features_r0.parquet' (format parquet)""")
print("build_sec", round(time.time()-t,2))
print(con.sql(f"select split, count(*), avg(y) from read_parquet('{ROOT}/data/features_r0.parquet') group by 1 order by 1").fetchall())
print(con.sql(f"select task, count(*) from read_parquet('{ROOT}/data/features_r0.parquet') group by 1 order by 2 desc").fetchall())
