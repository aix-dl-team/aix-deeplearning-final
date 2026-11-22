# 실험 코드와 결과

블로그([aix-dl-team.github.io/aix-deeplearning-final](https://aix-dl-team.github.io/aix-deeplearning-final/))에 나온 모든 숫자와 그래프를 만든 코드다.

## 준비

- 데이터: [AIDev-7.6M](https://huggingface.co/datasets/hao-li/AIDev-7.6M)의 AIDev-pop 테이블 4개(`pull_request`, `repository`, `pr_commits`, `pr_commit_details`, 합계 약 13.2GB)를 내려받는다. `scripts/common.py`의 `DL`을 내려받은 폴더로 바꾼다.
- 라이브러리: Python 3, duckdb 1.5, pandas 2.3, pyarrow, xgboost 3.2, scikit-learn 1.7, matplotlib, psutil
- 실험 환경: 메모리 약 4GB(3.9GB), 4코어 리눅스 가상 환경

## 실행 순서 (`scripts/` 안에서)

| 순서 | 명령 | 하는 일 | 결과 |
| --- | --- | --- | --- |
| 1 | `python3 eda_pr.py` | 에이전트별 PR 수·머지율, 월별 추이, 커밋 1개 비율, 스타 구간별 머지율 | 그래프 1~4, `results/eda_pr.json` |
| 2 | `python3 build_r0.py` | R0 피처(코드 안 봄)와 시간 분할 | `data/features_r0.parquet` |
| 3 | `python3 build_r1.py` | R1 피처(변경 통계, 테스트 파일 여부) | `data/features_r1.parquet` |
| 4 | `python3 eda_details.py` | PR 크기, 테스트 포함률, 테스트별 머지율 | 그래프 5~6, `results/eda_details.json` |
| 5 | `python3 run_seeds.py R0 R1 R0_single R1_single` | 같은 XGBoost를 시드 5개로 학습·평가 (시간 분할, 무작위 분할) | `results/models_seeds.json` |
| 6 | `python3 r2_extract.py` (끝날 때까지 반복) | 표본 PR 6만 개의 diff 텍스트 추출 | `data/diff/chunk_*.parquet` |
| 7 | `python3 r2_seeds.py` | 같은 표본에서 R0·R1·R2를 시드 5개로 비교 | `results/r2_seeds.json` |
| 8 | `python3 summarize_seeds.py` | 5번 평균·범위, 부트스트랩 95% 범위 | `results/summary_seeds.json` |
| 9 | `REPS=3 python3 bench.py pr_pandas_all pr_pandas_cols pr_duckdb` 와 `REPS=1 python3 bench.py cd_pandas_all cd_pandas_cols cd_duckdb` | pandas vs DuckDB 메모리·시간 (PR 테이블 3번, 파일 변경 테이블 1번 측정) | `results/bench.json` |
| 10 | `python3 fig_results.py` | 결과 그래프 7~9, PR 1,000개당 비용 | `results/cost_per_1k.json` |

- 블로그의 모델 성능 숫자는 `results/summary_seeds.json`(5번 반복 평균)에서 왔다.
- `results/models.json`은 `train_eval.py`로 한 번 학습한 결과이고, 피처 중요도(gain)를 볼 때 썼다.
- 표본 PR 목록 `data/sample_ids.parquet`(학습 기간 3만 개, 평가 기간 3만 개를 무작위로 뽑은 것)은 이 폴더에 함께 올렸다. 다른 피처 파일과 diff 조각은 용량 때문에 올리지 않았다(위 순서대로 실행하면 다시 만들어진다). 이 폴더에서 실행하려면 `code/data`에 원래 프로젝트의 `data/` 내용이 있어야 한다.
- 처리 시간은 같은 환경에서도 잴 때마다 흔들린다. `results/build_times.json`에 3번 잰 값을 모두 적었다.

## 그 밖의 폴더

- `api/`: 예측 API용 입력·출력 형식(`API_SPEC.md`), 학습과 똑같이 피처를 계산하는 `pr_features.py`, R1 모델 파일과 `predict.py`, 참고용 서버 예시. 백엔드 담당이 이 폴더를 가져다 쓴다.
- `r3/`: CodeBERT 임베딩(R3) 실험 노트북(Colab용)과 실행 가이드. 표본 데이터 `r3_sample.parquet`(약 79MB)은 용량 때문에 저장소에 올리지 않고 따로 전달한다.
