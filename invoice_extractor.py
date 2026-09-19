import os
import base64
import re

from dotenv import load_dotenv
from openai import OpenAI

from schemas import Transaction


# ==========================================
# OPENROUTER SETUP
# ==========================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ==========================================
# PLACEHOLDER SUPPLIER NAME CHECK
# ==========================================

def clean_supplier_name(name):
    if name is None:
        return None

    cleaned = " ".join(
        name.strip().lower().split()
    )

    placeholder_names = {
        "add company name",
        "add name",
        "company name",
        "add company",
        "supplier name",
        "your company name",
        "your company",
        "enter company name",
        "enter company",
        "enter supplier name",
        "company",
        "supplier"
    }

    if cleaned in placeholder_names:
        return None

    return name.strip()


# ==========================================
# DATE SAFETY
# ==========================================

def has_valid_date(value):
    if not value:
        return False

    pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    return bool(pattern.match(str(value)))


# ==========================================
# INVOICE EXTRACTION
# ==========================================

def extract_invoice(image_path: str) -> Transaction:

    # ==========================================
    # READ IMAGE
    # ==========================================

    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    # ==========================================
    # SHORT, FOCUSED PROMPT
    # ==========================================

    prompt = """
You are extracting data from an Indian GST/tax invoice.

Inspect the entire invoice image carefully.

Return ONLY valid JSON with exactly these fields:

{
  "transaction_type": "PURCHASE",
  "supplier_name": null,
  "amount": null,
  "payment_status": null,
  "transaction_date": null,
  "reference_number": null,
  "notes": null
}

Rules:

1. transaction_type:
   Use PURCHASE for a normal sales/purchase invoice.
   Other allowed values are PAYMENT, RETURN, CREDIT_NOTE.

2. supplier_name:
   Extract the REAL company that issued the invoice.
   Do NOT use the customer/buyer name.
   Do NOT use placeholder text such as:
   "Add Company Name", "Company Name", "Supplier Name",
   "Your Company", "Enter Company Name".
   If unclear, use null.

3. amount:
   Use the FINAL invoice total / grand total / total payable.
   Do NOT use subtotal, item amount, tax amount or discount.
   Return only the numeric value.

4. transaction_date:
   Use the INVOICE DATE only.
   Do NOT use due date, delivery date or e-way bill date.
   Convert to YYYY-MM-DD.
   If unclear, use null.

5. reference_number:
   Use the INVOICE NUMBER.
   Do NOT use IRN, Ack No. or e-way bill number.
   If unclear, use null.

6. payment_status:
   Extract only if clearly stated.
   Otherwise null.

7. notes:
   Only include useful additional invoice information.
   Otherwise null.

8. Never invent information.
9. Do not calculate balances.
10. Do not modify any database.
11. Understand Indian invoices, GST, INR and English/Hindi/Hinglish.

IMPORTANT:
The final total is the amount to extract.
The invoice date is the transaction date.
The seller/issuer is the supplier.
"""


    # ==========================================
    # FREE OPENROUTER MODEL
    # ==========================================

    response = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=180,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{image_data}"
                            )
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        }
    )


    # ==========================================
    # GET AI RESPONSE
    # ==========================================

    data = response.choices[0].message.content

    if not data:
        raise ValueError(
            "Invoice extractor returned empty response"
        )


    # ==========================================
    # JSON → TRANSACTION
    # ==========================================

    transaction = Transaction.model_validate_json(
        data
    )


    # ==========================================
    # SUPPLIER SAFETY
    # ==========================================

    transaction.supplier_name = clean_supplier_name(
        transaction.supplier_name
    )


    # ==========================================
    # DATE SAFETY
    # ==========================================

    if transaction.transaction_date is not None:

        if not has_valid_date(
            transaction.transaction_date
        ):
            transaction.transaction_date = None


    # ==========================================
    # AMOUNT SAFETY
    # ==========================================

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None


    return transaction