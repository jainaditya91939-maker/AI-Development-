from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from final_intelligence import generate_final_intelligence
from processor import process_transaction, process_invoice

import os
import shutil


app = FastAPI(
    title="AI Business Investigator AI Service"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://192.168.0.106:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==============================
# REQUEST SCHEMAS
# ==============================

class InvestigatorRequest(BaseModel):
    question: str


class VoiceTransactionRequest(BaseModel):
    text: str


# ==============================
# HOME
# ==============================

@app.get("/")
def home():
    return {
        "message": "AI Business Investigator AI Service is running!"
    }


# ==============================
# COMPLETE BUSINESS INTELLIGENCE
# ==============================

@app.get("/api/v1/ai/investigate")
def investigate():

    result = generate_final_intelligence()

    return {
        "status": "SUCCESS",
        "report": result
    }


# ==============================
# QUESTION-BASED AI INVESTIGATOR
# ==============================

@app.post("/api/v1/ai/investigate")
def investigate_question(request: InvestigatorRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    result = generate_final_intelligence(
        question=request.question
    )

    return {
        "status": "SUCCESS",
        "question": request.question,
        "answer": result
    }


# ==============================
# VOICE / TEXT TRANSACTION
# ==============================

@app.post("/api/v1/ai/voice/transaction")
def voice_transaction(request: VoiceTransactionRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Voice text cannot be empty"
        )

    result = process_transaction(
        request.text
    )

    return result


# ==============================
# INVOICE TRANSACTION
# ==============================

@app.post("/api/v1/ai/invoice/transaction")
def invoice_transaction(
    file: UploadFile = File(...)
):

    # Check file type
    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/pdf"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG or PDF invoice files are allowed"
        )

    # Temporary directory
    temp_dir = "temp_invoices"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    # Create safe file path
    file_path = os.path.join(
        temp_dir,
        file.filename
    )

    try:

        # Save uploaded invoice
        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # Process invoice
        result = process_invoice(
            file_path
        )

        return result

    except Exception as e:

        print(
            "Invoice processing error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="Invoice processing failed"
        )

    finally:

        # Delete temporary invoice
        if os.path.exists(file_path):

            os.remove(
                file_path
            )