"""R1: 코드 변경 통계 (줄 수, 파일 수, 확장자, 테스트 파일 여부). patch 본문은 읽지 않는다."""
import duckdb, time, json
from common import *
TEST_RE = r"(^|/)(tests?|__tests__|specs?|testing)/|(^|/)test_[^/]*$|_test\.[a-z0-9]+$|\.(test|spec)\.[a-z0-9]+$|tests?\.(java|cs|kt|swift)$"
DOC_RE = r"\.(md|mdx|rst|txt|adoc)$|(^|/)docs?/"
t = time.time()
con = duckdb.connect(); con.sql("set memory_limit='1.5GB'"); con.sql("set threads=4"); con.sql("set enable_progress_bar=false"); con.sql("set temp_directory='/tmp/duck'")
con.sql(f"""copy (
 with d as (select pr_id, sha, lower(filename) as f, additions, deletions from {pq('pr_commit_details')}),
 f as (select *, regexp_matches(f, '{TEST_RE}') as is_test, regexp_matches(f, '{DOC_RE}') as is_doc,
              regexp_extract(f, '\\.([a-z0-9]+)$', 1) as ext from d)
 select pr_id,
   count(distinct sha) as n_commits,
   count(distinct f) as n_files, sum(case when f is null then 1 else 0 end) as n_null_files,
   sum(additions) as additions, sum(deletions) as deletions,
   ln(1 + sum(additions) + sum(deletions)) as log_churn,
   count(distinct ext) as n_ext,
   mode(ext) as top_ext,
   coalesce(max(is_test::int), 0) as has_test,
   coalesce(sum(case when is_test then additions + deletions else 0 end) / nullif(sum(additions + deletions), 0), 0) as test_line_share,
   sum(case when is_doc then 1 else 0 end) * 1.0 / count(*) as doc_file_share,
   bool_and(is_test) as test_only
 from f group by pr_id
) to '{ROOT}/data/features_r1.parquet' (format parquet)""")
sec = time.time() - t
print("build_sec", round(sec, 1))
json.dump({"build_r1_sec": sec}, open(f"{RES}/build_r1_time.json", "w"))
print(con.sql(f"select count(*), avg(has_test), avg(case when n_commits=1 then 1.0 else 0 end), sum(case when n_files=0 then 1 else 0 end) from read_parquet('{ROOT}/data/features_r1.parquet')").fetchall())
