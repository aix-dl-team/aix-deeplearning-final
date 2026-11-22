"""PR 정보(JSON)를 받아 학습 때와 똑같은 방법으로 R0·R1 피처를 계산한다.

학습 피처는 scripts/build_r0.py, scripts/build_r1.py(DuckDB SQL)로 만들었다.
이 파일은 같은 규칙을 파이썬으로 옮긴 것이고, test_pr_features.py로 학습 피처 파일과 값이 같은지 확인했다.
"""
import math, re
from collections import Counter

# build_r1.py와 같은 정규식 (파일 이름은 소문자로 바꾼 뒤 검사)
TEST_RE = re.compile(r"(^|/)(tests?|__tests__|specs?|testing)/|(^|/)test_[^/]*$|_test\.[a-z0-9]+$|\.(test|spec)\.[a-z0-9]+$|tests?\.(java|cs|kt|swift)$")
DOC_RE = re.compile(r"\.(md|mdx|rst|txt|adoc)$|(^|/)docs?/")
EXT_RE = re.compile(r"\.([a-z0-9]+)$")
# build_r0.py와 같은 작업 종류 규칙 (위에서부터 먼저 맞는 것)
TASK_RULES = [("feat", r"^(feat|feature)"), ("fix", r"^(fix|bug|hotfix)"), ("docs", r"^docs?"), ("test", r"^tests?"),
              ("refactor", r"^refactor"), ("chore", r"^(chore|build|ci|deps)"), ("perf", r"^perf")]
AGENTS = {"Claude_Code", "Copilot", "OpenAI_Codex", "Cursor", "Devin", "Google_Jules"}

R0_CAT = ["agent", "lang", "task"]
R0_NUM = ["stars", "forks", "repo_prior_prs", "title_len", "body_len"]
R1_CAT = R0_CAT + ["top_ext"]
R1_NUM = R0_NUM + ["n_files", "additions", "deletions", "log_churn", "n_ext", "has_test", "test_line_share", "doc_file_share"]


def task_of(title):
    t = (title or "").lower()
    for name, pat in TASK_RULES:
        if re.search(pat, t):
            return name
    return "other"


def r0_features(pr):
    repo = pr.get("repo") or {}
    return {
        "agent": pr["agent"],
        "lang": repo.get("language") or "unknown",
        "task": task_of(pr.get("title")),
        "stars": int(repo.get("stars") or 0),
        "forks": int(repo.get("forks") or 0),
        "repo_prior_prs": int(pr.get("repo_prior_prs") or 0),
        "title_len": len(pr.get("title") or ""),
        "body_len": len(pr.get("body") or ""),
    }


def _rows(pr):
    """커밋별 파일 변경을 한 줄씩 펼친다. 학습 데이터(pr_commit_details)와 같은 단위다."""
    if pr.get("commits"):
        for c in pr["commits"]:
            for f in c.get("files", []):
                yield f
    else:  # 커밋 구분 없이 파일 목록만 온 경우: 커밋 1개짜리 PR로 본다
        for f in pr.get("files", []):
            yield f


def r1_features(pr):
    names, exts, adds, dels = [], [], 0, 0
    n_rows = n_doc = 0
    any_add = any_del = False
    test_lines, all_lines, has_test = 0, 0, 0
    for f in _rows(pr):
        n_rows += 1
        fn = f.get("filename")
        a, d = f.get("additions"), f.get("deletions")
        if a is not None: adds += a; any_add = True
        if d is not None: dels += d; any_del = True
        # SQL처럼: 추가·삭제 중 하나라도 비어 있으면 그 줄의 합계는 계산에서 빠진다
        line = a + d if (a is not None and d is not None) else None
        if line is not None: all_lines += line
        if fn is None:
            continue
        fl = fn.lower()
        names.append(fl)
        m = EXT_RE.search(fl)
        exts.append(m.group(1) if m else "")  # DuckDB regexp_extract는 못 찾으면 빈 문자열
        if TEST_RE.search(fl):
            has_test = 1
            if line is not None: test_lines += line
        if DOC_RE.search(fl):
            n_doc += 1
    if n_rows == 0:
        raise ValueError("파일 변경 정보가 없는 PR은 R1 피처를 만들 수 없다 (학습에서도 제외했다)")
    ext_counts = Counter(exts)
    top_ext = max(ext_counts.items(), key=lambda kv: kv[1])[0] if ext_counts else None
    # 줄 수 정보가 전부 비어 있으면 학습 데이터처럼 '값 없음'(None)으로 둔다. XGBoost가 빈 값으로 처리한다.
    a_sum = adds if any_add else None; d_sum = dels if any_del else None
    return {
        "n_files": len(set(names)),
        "additions": a_sum, "deletions": d_sum,
        "log_churn": math.log(1 + a_sum + d_sum) if (a_sum is not None and d_sum is not None) else None,
        "n_ext": len(ext_counts),
        "top_ext": top_ext,
        "has_test": has_test,
        "test_line_share": (test_lines / all_lines) if all_lines else 0.0,
        "doc_file_share": n_doc / n_rows,
    }


def build_features(pr):
    """API 입력 JSON 하나 → R1 모델 입력 피처 17개 (dict)."""
    if pr.get("agent") not in AGENTS:
        raise ValueError(f"agent는 {sorted(AGENTS)} 중 하나여야 한다")
    return {**r0_features(pr), **r1_features(pr)}
