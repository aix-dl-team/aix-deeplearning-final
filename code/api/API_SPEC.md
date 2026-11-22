# PR 머지 예측 API: 입력·출력 형식 (초안 v1)

백엔드 담당(전성민)과 모델 담당(정하랑)이 맞춰 쓰는 약속이다. 바꾸고 싶은 부분이 있으면 이 문서를 고친 뒤 서로 확인한다.

## 한눈에 보기

- 모델: **R1** (겉정보 + 바뀐 코드 통계, 피처 17개, XGBoost). 실험에서 비용 대비 성능이 가장 좋았다.
- 성능: 과거 PR로 학습하고 미래 PR로 평가했을 때 AUC 약 0.727 (5번 평균)
- 실제로 쓸 모델은 정답이 있는 닫힌 PR 331,008개 전부(2024-12-24 ~ 2026-03-31)로 다시 학습했다. 모델 버전 `r1-xgb-20260331`
- 처리 시간: 요청 하나에 약 3~7ms (피처 계산 포함, 4코어 가상 환경에서 측정)

## 폴더 구성

| 파일 | 내용 |
| --- | --- |
| `pr_features.py` | 입력 JSON → 피처 17개. 학습 코드(DuckDB SQL)와 같은 규칙. **피처는 반드시 이 함수로만 계산한다** |
| `predict.py` | `MergePredictor` 클래스. 모델을 불러오고 `predict(pr_json)`으로 결과를 돌려준다 |
| `model_r1.json`, `model_meta.json` | 모델 파일과 설명(피처 순서, 범주 목록, 학습 기간). 둘은 항상 같이 다닌다 |
| `example_request.json` | 요청 예시 |
| `server_example.py` | 참고용 최소 FastAPI 서버 (`/health`, `/predict`) |
| `test_pr_features.py` | 학습 피처 파일과 값이 같은지 확인하는 테스트 (원본 데이터 필요) |
| `train_api_model.py` | 모델 파일을 다시 만드는 코드 |

빠른 확인: `pip install xgboost==3.2.0 pandas numpy` 후 `python3 predict.py example_request.json`

## 요청 (POST /predict)

```json
{
  "agent": "Claude_Code",
  "title": "fix: handle empty config file",
  "body": "빈 설정 파일을 읽을 때 생기는 오류를 고쳤습니다.",
  "repo": {"full_name": "example-org/example-repo", "stars": 350, "forks": 40, "language": "Python"},
  "repo_prior_prs": 5,
  "commits": [
    {"sha": "a1b2c3d", "files": [
      {"filename": "src/config.py", "additions": 12, "deletions": 3},
      {"filename": "tests/test_config.py", "additions": 25, "deletions": 0}
    ]}
  ]
}
```

| 필드 | 필수 | 설명 |
| --- | --- | --- |
| `agent` | 예 | `Claude_Code`, `Copilot`, `OpenAI_Codex`, `Cursor`, `Devin`, `Google_Jules` 중 하나 (학습한 6종만 가능) |
| `title`, `body` | 아니오 | PR 제목과 본문. 없으면 빈 글자로 본다. 제목 첫 단어로 작업 종류(fix, feat 등)를 정한다 |
| `repo.stars`, `repo.forks` | 아니오 | 저장소 별·포크 수. 없으면 0 |
| `repo.language` | 아니오 | 저장소 주 언어. 없으면 `unknown` |
| `repo.full_name` | 아니오 | 모델은 안 쓴다. 저장·로그용 |
| `repo_prior_prs` | 아니오 | 같은 저장소에 이 PR보다 **먼저** 올라온 AI 에이전트 PR 수. 없으면 0 (아래 주의 참고) |
| `commits` | `commits`나 `files` 중 하나는 필수 | 커밋마다 바뀐 파일 목록. 파일마다 `filename`, `additions`(추가 줄 수), `deletions`(삭제 줄 수) |
| `files` | 위와 같음 | 커밋 구분 없이 파일 목록만 줄 때. 커밋 1개짜리 PR로 보고 계산한다 |

## 응답

성공 (200):

```json
{"merge_probability": 0.9165, "model_version": "r1-xgb-20260331", "elapsed_ms": 4.2}
```

- `merge_probability`: 머지될 확률 (0~1, 소수 넷째 자리). 낮을수록 먼저 검토할 후보다
- `model_version`: 어떤 모델이 계산했는지. 결과를 저장할 때 꼭 같이 남긴다
- `elapsed_ms`: 피처 계산 + 예측에 걸린 시간 (밀리초)

오류 (공통 형식 `{"error": {"code": "...", "message": "..."}}`):

| 상태 코드 | `code` | 언제 |
| --- | --- | --- |
| 400 | `bad_request` | JSON이 아니거나 `agent`, `commits`/`files`가 없음 |
| 422 | `invalid_pr` | 학습하지 않은 에이전트, 파일 변경 정보가 하나도 없는 PR (학습에서도 뺐다) |
| 500 | `model_error` | 그 밖의 예측 오류 |

## 결과 저장 (제안)

요청 하나마다 아래를 저장해 두면, 나중에 PR이 실제로 머지됐는지 붙여서 **실제 서비스에서의 정확도**를 잴 수 있다.

| 칸 | 예 |
| --- | --- |
| 요청 시각, 저장소, PR 번호(있으면) | 2026-12-01T10:00:00+09:00, example-org/example-repo, 123 |
| 요청 원본 JSON | (그대로) |
| 계산된 피처 17개 | `pr_features.build_features()` 결과 |
| 머지 확률, 모델 버전, 처리 시간 | 0.9165, r1-xgb-20260331, 4.2 |
| 실제 결과 (나중에 채움) | merged / closed / open |

## 꼭 지킬 것

1. **피처는 `pr_features.build_features()`로만 계산한다.** 학습할 때와 계산이 조금만 달라도 모델이 엉뚱한 확률을 낸다. 이 함수는 학습 피처 파일과 무작위 PR 2,989개에서 17개 값이 모두 같은 것을 확인했다.
2. **커밋별 파일 목록(`commits`)을 주는 게 정확하다.** 학습 데이터는 "커밋마다 바뀐 파일"을 모두 더한 값이다. GitHub의 PR 파일 목록(최종 변경만 보여 줌)을 `files`로 주면, 커밋이 여러 개인 PR에서는 줄 수가 학습 때와 달라질 수 있다.
3. **모델은 서버 시작할 때 한 번만 불러온다.** 요청마다 불러오면 느리다.

## 알려진 한계

- `repo_prior_prs`: 학습 때는 데이터셋 안에서 같은 저장소의 이전 AI PR 수를 셌다. 서버에서는 저장해 둔 과거 요청으로 세거나, 모르면 0을 넣는다. 0을 넣으면 실제보다 "처음 보는 저장소"처럼 계산된다.
- 별·포크 수: 학습 데이터는 데이터를 모은 시점(2026년 3월)의 값이다. 서버는 요청 시점 값을 받으므로 조금 차이가 날 수 있다.
- 에이전트 6종 말고는 예측하지 않는다.

## 나중에 (시간 나면)

- GitHub PR 주소만 받으면 정보를 자동으로 모으기: GitHub REST API로 PR 정보(`GET /repos/{owner}/{repo}/pulls/{number}`), 커밋 목록(`.../pulls/{number}/commits`), 커밋별 파일(`GET /repos/{owner}/{repo}/commits/{sha}`)을 받아 위 요청 형식으로 바꾸면 된다.
