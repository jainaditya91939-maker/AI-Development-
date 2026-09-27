import os
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from final_intelligence import generate_final_intelligence
from processor import process_transaction, process_invoice


app = FastAPI(title="AI Business Investigator AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-business-investigator-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=True)


class InvestigatorRequest(BaseModel):
    question: str


class VoiceTransactionRequest(BaseModel):
    text: str


def get_token(credentials: HTTPAuthorizationCredentials) -> str:
    return credentials.credentials


@app.get("/")
def home():
    return {"message": "AI Business Investigator AI Service is running!"}


@app.get("/api/v1/ai/investigate")
def investigate(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = get_token(credentials)
    result = generate_final_intelligence(token=token)
    return {"status": "SUCCESS", "report": result}


@app.post("/api/v1/ai/investigate")
def investigate_question(
    request: InvestigatorRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    token = get_token(credentials)
    result = generate_final_intelligence(question=request.question, token=token)
    return {
        "status": "SUCCESS",
        "question": request.question,
        "answer": result,
    }


@app.post("/api/v1/ai/voice/transaction")
def voice_transaction(
    request: VoiceTransactionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Voice text cannot be empty")

    token = get_token(credentials)
    return process_transaction(request.text, token)


@app.post("/api/v1/ai/invoice/transaction")
def invoice_transaction(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG or WEBP invoice images are allowed",
        )

    suffix = Path(file.filename or "invoice.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        token = get_token(credentials)
        return process_invoice(temp_path, token)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Invoice processing failed")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
