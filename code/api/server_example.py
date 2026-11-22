"""참고용 최소 API 서버 예시 (FastAPI). 실제 서버 구조, 저장소(DB), 로그는 백엔드 담당이 정한다.
실행: pip install fastapi uvicorn → uvicorn server_example:app --reload"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from predict import MergePredictor

app = FastAPI(title="PR merge predictor")
model = MergePredictor()  # 서버 시작할 때 한 번만 불러온다


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model.meta["model_version"]}


@app.post("/predict")
def predict(pr: dict):
    if not isinstance(pr.get("agent"), str) or not (pr.get("commits") or pr.get("files")):
        return JSONResponse(status_code=400, content={"error": {"code": "bad_request", "message": "agent와 commits(또는 files)는 꼭 있어야 한다"}})
    try:
        return model.predict(pr)
    except ValueError as e:  # 지원하지 않는 에이전트, 파일 변경 정보 없음 등
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_pr", "message": str(e)}})
    except Exception:
        return JSONResponse(status_code=500, content={"error": {"code": "model_error", "message": "예측 중 오류가 났다"}})
