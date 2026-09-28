import os
import base64
import json
import mimetypes
import re

from dotenv import load_dotenv
from openai import OpenAI
from schemas import Transaction


load_dotenv()


# ============================================================
# OPENROUTER CONFIG
# ============================================================

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not configured")


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# MODEL
# ============================================================

MODEL = "google/gemma-4-31b-it:free"


# ============================================================
# SUPPLIER CLEANING
# ============================================================

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
        "supplier",
    }

    if cleaned in placeholder_names:
        return None

    return name.strip()


# ============================================================
# DATE VALIDATION
# ============================================================

def has_valid_date(value):
    if not value:
        return False

    return bool(
        re.match(
            r"^\d{4}-\d{2}-\d{2}$",
            str(value),
        )
    )


# ============================================================
# MIME TYPE
# ============================================================

def get_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        return mime_type

    return "image/png"


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Direct JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # JSON object inside extra text
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match:
        candidate = match.group(0)

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# NORMALIZE TRANSACTION DATA
# ============================================================

def normalize_transaction_data(data):

    if not isinstance(data, dict):
        raise ValueError(
            "Invoice AI did not return a JSON object"
        )

    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    # Invoice extraction is always treated as PURCHASE.
    data["transaction_type"] = "PURCHASE"

    # --------------------------------------------------------
    # Supplier
    # --------------------------------------------------------

    data["supplier_name"] = clean_supplier_name(
        data.get("supplier_name")
    )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    amount = data.get("amount")

    if amount is not None:

        try:

            if isinstance(amount, str):

                amount = (
                    amount
                    .replace("₹", "")
                    .replace(",", "")
                    .replace("INR", "")
                    .strip()
                )

            amount = float(amount)

            if amount <= 0:
                amount = None

        except (ValueError, TypeError):

            amount = None

    data["amount"] = amount

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    transaction_date = data.get(
        "transaction_date"
    )

    if transaction_date:

        transaction_date = str(
            transaction_date
        ).strip()

        if not has_valid_date(
            transaction_date
        ):
            transaction_date = None

    data["transaction_date"] = transaction_date

    # --------------------------------------------------------
    # Reference number
    # --------------------------------------------------------

    reference_number = data.get(
        "reference_number"
    )

    if reference_number is not None:

        reference_number = str(
            reference_number
        ).strip()

        if not reference_number:
            reference_number = None

    data["reference_number"] = reference_number

    # --------------------------------------------------------
    # Payment status
    # --------------------------------------------------------

    payment_status = data.get(
        "payment_status"
    )

    if payment_status is not None:

        payment_status = str(
            payment_status
        ).strip()

        if not payment_status:
            payment_status = None

    data["payment_status"] = payment_status

    # --------------------------------------------------------
    # Notes
    # --------------------------------------------------------

    notes = data.get("notes")

    if notes is not None:

        notes = str(notes).strip()

        if not notes:
            notes = None

    data["notes"] = notes

    return data


# ============================================================
# MAIN INVOICE EXTRACTION
# ============================================================

def extract_invoice(image_path: str) -> Transaction:

    print(
        "========================================",
        flush=True,
    )

    print(
        "INVOICE EXTRACTION STARTED",
        flush=True,
    )

    print(
        "MODEL:",
        MODEL,
        flush=True,
    )

    print(
        "IMAGE:",
        image_path,
        flush=True,
    )

    print(
        "OPENROUTER_API_KEY PRESENT:",
        bool(api_key),
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "rb",
        ) as image_file:

            image_data = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    except Exception as e:

        print(
            "INVOICE IMAGE READ ERROR:",
            repr(e),
            flush=True,
        )

        raise ValueError(
            f"Could not read invoice image: {e}"
        )

    mime_type = get_mime_type(
        image_path
    )

    print(
        "INVOICE MIME TYPE:",
        mime_type,
        flush=True,
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = """
Read this invoice image carefully.

Return ONLY valid JSON.

Do not write anything before or after the JSON.

Use exactly this format:

{
  "transaction_type": "PURCHASE",
  "supplier_name": null,
  "amount": null,
  "payment_status": null,
  "transaction_date": null,
  "reference_number": null,
  "notes": null
}

EXTRACTION RULES:

1. supplier_name

Use the seller/company that issued the invoice.

Do NOT use the buyer/customer.

If the seller/company cannot be identified, use null.

2. amount

Use the final GRAND TOTAL or TOTAL PAYABLE.

Do NOT use subtotal.

Do NOT use tax-only amount.

Do NOT use an individual product amount.

Return only the numeric amount.

For example:

25706

not:

"₹25,706"

3. transaction_date

Use the invoice date.

Convert it to:

YYYY-MM-DD

If the date cannot be identified, use null.

4. reference_number

Use the invoice number.

Do NOT use:

- IRN
- Ack number
- Acknowledgement number
- E-way bill number

5. transaction_type

For a normal purchase/sales invoice, ALWAYS return:

"PURCHASE"

6. payment_status

Only use this if the invoice explicitly states something such as:

PAID
UNPAID
PENDING
PARTIALLY PAID

Otherwise return null.

7. notes

Use null unless useful information is clearly visible.

8. Never invent information.

If something cannot be read, use null.

IMPORTANT:

Return ONLY JSON.

Example:

{
  "transaction_type": "PURCHASE",
  "supplier_name": "Havells India Ltd",
  "amount": 25706,
  "payment_status": null,
  "transaction_date": "2026-09-28",
  "reference_number": "INV-001",
  "notes": null
}
"""

    # --------------------------------------------------------
    # OpenRouter request
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=700,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};"
                                    f"base64,{image_data}"
                                )
                            },
                        },
                    ],
                }
            ],
        )

    except Exception as e:

        print(
            "========================================",
            flush=True,
        )

        print(
            "OPENROUTER INVOICE ERROR",
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

        print(
            "========================================",
            flush=True,
        )

        raise ValueError(
            f"Invoice AI request failed: {e}"
        )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not response.choices:

        raise ValueError(
            "Invoice AI returned no choices"
        )

    choice = response.choices[0]

    print(
        "INVOICE MODEL:",
        MODEL,
        flush=True,
    )

    print(
        "INVOICE FINISH:",
        choice.finish_reason,
        flush=True,
    )

    data = choice.message.content

    print(
        "INVOICE RAW CONTENT:",
        repr(data),
        flush=True,
    )

    if not data:

        raise ValueError(
            "Invoice AI returned empty content"
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    raw_data = extract_json(data)

    if raw_data is None:

        raise ValueError(
            "Invoice AI returned invalid JSON"
        )

    print(
        "INVOICE PARSED JSON:",
        raw_data,
        flush=True,
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    raw_data = normalize_transaction_data(
        raw_data
    )

    print(
        "INVOICE NORMALIZED DATA:",
        raw_data,
        flush=True,
    )

    # --------------------------------------------------------
    # Pydantic validation
    # --------------------------------------------------------

    try:

        transaction = Transaction.model_validate(
            raw_data
        )

    except Exception as e:

        print(
            "INVOICE SCHEMA VALIDATION ERROR:",
            repr(e),
            flush=True,
        )

        raise ValueError(
            f"Invoice data validation failed: {e}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    if transaction.transaction_date:

        if not has_valid_date(
            transaction.transaction_date
        ):
            transaction.transaction_date = None

    print(
        "========================================",
        flush=True,
    )

    print(
        "INVOICE EXTRACTION SUCCESS",
        flush=True,
    )

    print(
        "TRANSACTION:",
        transaction.model_dump(
            mode="json"
        ),
        flush=True,
    )

    print(
        "========================================",
        flush=True,
    )

    return transaction