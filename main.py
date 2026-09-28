import os
import shutil
import tempfile
import traceback
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Business Investigator AI Service"
)


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# AUTH
# ============================================================

security = HTTPBearer(auto_error=True)


def get_token(
    credentials: HTTPAuthorizationCredentials,
) -> str:
    return credentials.credentials


# ============================================================
# REQUEST MODELS
# ============================================================

class InvestigatorRequest(BaseModel):
    question: str


class VoiceTransactionRequest(BaseModel):
    text: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Business Investigator AI Service is running!"
    }


# ============================================================
# AI INVESTIGATOR - GET
# ============================================================

@app.get("/api/v1/ai/investigate")
def investigate(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        # Lazy import
        from final_intelligence import (
            generate_final_intelligence
        )

        token = get_token(credentials)

        result = generate_final_intelligence(
            token=token
        )

        return {
            "status": "SUCCESS",
            "report": result,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "========== INVESTIGATE ERROR ==========",
            flush=True,
        )
        print(
            "ERROR TYPE:",
            type(e).__name__,
            flush=True,
        )
        print(
            "ERROR:",
            repr(e),
            flush=True,
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"AI investigation failed: {str(e)}",
        )


# ============================================================
# AI INVESTIGATOR - POST
# ============================================================

@app.post("/api/v1/ai/investigate")
def investigate_question(
    request: InvestigatorRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        # Lazy import
        from final_intelligence import (
            generate_final_intelligence
        )

        token = get_token(credentials)

        result = generate_final_intelligence(
            question=request.question,
            token=token,
        )

        return {
            "status": "SUCCESS",
            "question": request.question,
            "answer": result,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "========== INVESTIGATE QUESTION ERROR ==========",
            flush=True,
        )
        print(
            "ERROR TYPE:",
            type(e).__name__,
            flush=True,
        )
        print(
            "ERROR:",
            repr(e),
            flush=True,
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"AI investigation failed: {str(e)}",
        )


# ============================================================
# VOICE TRANSACTION
# ============================================================

@app.post("/api/v1/ai/voice/transaction")
def voice_transaction(
    request: VoiceTransactionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Voice text cannot be empty",
        )

    try:
        print(
            "========== VOICE REQUEST ==========",
            flush=True,
        )

        # Lazy import
        from processor import process_transaction

        token = get_token(credentials)

        print(
            "Voice text:",
            request.text,
            flush=True,
        )

        result = process_transaction(
            request.text,
            token,
        )

        print(
            "Voice result:",
            repr(result),
            flush=True,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        print(
            "========== VOICE TRANSACTION ERROR ==========",
            flush=True,
        )
        print(
            "ERROR TYPE:",
            type(e).__name__,
            flush=True,
        )
        print(
            "ERROR:",
            repr(e),
            flush=True,
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Voice transaction processing failed: {str(e)}",
        )


# ============================================================
# INVOICE TRANSACTION
# ============================================================

@app.post("/api/v1/ai/invoice/transaction")
def invoice_transaction(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    print(
        "🔥🔥🔥 INVOICE ENDPOINT HIT 🔥🔥🔥",
        flush=True,
    )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG or WEBP invoice images are allowed",
        )

    suffix = Path(
        file.filename or "invoice.jpg"
    ).suffix.lower()

    if suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        suffix = ".jpg"

    temp_path = None

    try:
        print(
            "Invoice filename:",
            file.filename,
            flush=True,
        )

        print(
            "Invoice content type:",
            file.content_type,
            flush=True,
        )

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_path = temp_file.name

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

        print(
            "Invoice saved:",
            temp_path,
            flush=True,
        )

        # ----------------------------------------------------
        # Get token
        # ----------------------------------------------------

        token = get_token(credentials)

        print(
            "Token received.",
            flush=True,
        )

        # ----------------------------------------------------
        # Lazy import
        # ----------------------------------------------------

        print(
            "Loading processor...",
            flush=True,
        )

        from processor import process_invoice

        print(
            "Processor loaded.",
            flush=True,
        )

        # ----------------------------------------------------
        # Process invoice
        # ----------------------------------------------------

        print(
            "Starting process_invoice...",
            flush=True,
        )

        result = process_invoice(
            temp_path,
            token,
        )

        print(
            "Invoice result:",
            repr(result),
            flush=True,
        )

        print(
            "🔥🔥🔥 INVOICE SUCCESS 🔥🔥🔥",
            flush=True,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        print(
            "🔥🔥🔥 INVOICE PROCESSING ERROR 🔥🔥🔥",
            flush=True,
        )

        print(
            "ERROR TYPE:",
            type(e).__name__,
            flush=True,
        )

        print(
            "ERROR:",
            repr(e),
            flush=True,
        )

        traceback.print_exc()

        print(
            "🔥🔥🔥 END INVOICE ERROR 🔥🔥🔥",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Invoice processing failed: {str(e)}",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)

                print(
                    "Temporary invoice removed.",
                    flush=True,
                )

            except Exception as e:
                print(
                    "Temporary file cleanup error:",
                    repr(e),
                    flush=True,
                )