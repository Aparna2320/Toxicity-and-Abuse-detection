from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.classifier import predict_toxicity, reload_config

app = FastAPI(
    title="Toxicity Detector API",
    version="1.0.0",
    description="Detect toxic or safe text with a configurable ML model."
)

# --- Input schema ---
class PostInput(BaseModel):
    user_id: str
    post_id: str
    text: str

# --- Output schema ---
class PredictionOutput(BaseModel):
    user_id: str
    post_id: str
    toxicity_score: float
    label: str
    action: str
    reasons: list
    threshold: float

# --- Exception handler (optional) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": exc.errors()}
    )

# --- Health check ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Version info ---
@app.get("/version")
def get_version():
    return {"version": "1.0.0"}

# --- Main toxicity check ---
@app.post("/analyze-text", response_model=PredictionOutput)
def analyze_text(input: PostInput):
    result = predict_toxicity(input.text)
    return {
        "user_id": input.user_id,
        "post_id": input.post_id,
        **result
    }

# --- Reload config without restarting ---
@app.post("/reload-config")
def reload():
    reload_config()
    return {"status": "config reloaded"}
