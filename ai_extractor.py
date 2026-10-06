import os
import re
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from schemas import Transaction


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = None

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ============================================================
# DATE DETECTION
# ============================================================

def has_explicit_date(text: str) -> bool:
    pattern = re.compile(
        r"""
        (
            \b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{4}\b
            |
            \b\d{1,2}\s+
            (?:
                january|february|march|april|may|june|
                july|august|september|october|november|december|
                jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec
            )
            \s+\d{4}\b
            |
            \b
            (?:
                january|february|march|april|may|june|
                july|august|september|october|november|december|
                jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec
            )
            \s+\d{1,2},?\s+\d{4}\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    return bool(pattern.search(text))


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amount(text: str):
    patterns = [
        r"₹\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        r"(?:rs\.?|rupees?)\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*rupees?",
        r"(?:₹\s*)?([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:ka|ki|ke)(?=\s|$|[.,!?])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if not match:
            continue

        value = match.group(1).replace(",", "")

        try:
            amount = float(value)

            if amount > 0:
                return amount

        except ValueError:
            pass

    return None


# ============================================================
# TRANSACTION TYPE
# ============================================================

def extract_transaction_type(text: str):
    value = text.lower().strip()

    # Credit note FIRST
    credit_words = [
        "credit note",
        "credit_note",
        "credit",
    ]

    for word in credit_words:
        if word in value:
            return "CREDIT_NOTE"

    # Return
    return_words = [
        "return",
        "returned",
        "return hua",
        "return hui",
        "return ki",
        "return kiya",
    ]

    for word in return_words:
        if word in value:
            return "RETURN"

    # Payment
    payment_words = [
        "payment",
        "paid",
        "pay",
        "payment ki",
        "payment hua",
        "payment kiya",
        "pay kiya",
        "payment kar",
    ]

    for word in payment_words:
        if word in value:
            return "PAYMENT"

    # Purchase
    purchase_words = [
        "purchase",
        "purchased",
        "buy",
        "bought",
        "purchase hua",
        "purchase hui",
        "purchase ki",
        "purchase kiya",
        "purchase kar",
        "maal liya",
        "maal liya hai",
        "samaan liya",
        "saman liya",
        "samaan kharida",
        "saman kharida",
    ]

    for word in purchase_words:
        if word in value:
            return "PURCHASE"

    return None


# ============================================================
# SUPPLIER NAME
# ============================================================

def extract_supplier_name(text: str):
    if not text:
        return None

    value = text.strip()

    patterns = [
        # Havells purchase 2000
        r"^(.+?)\s+(?:purchase|purchased|buy|bought)\b",

        # Havells payment 500
        r"^(.+?)\s+(?:payment|paid|pay)\b",

        # Havells return 1000
        r"^(.+?)\s+(?:return|returned)\b",

        # Havells credit note 300
        r"^(.+?)\s+credit\s+note\b",

        # Havells 500 purchase
        r"^(.+?)\s+(?:₹|rs\.?|rupees?)\s*[0-9][0-9,]*(?:\.[0-9]+)?\s+(?:purchase|payment|return|credit)\b",

        # Havells se 500
        r"^(.+?)\s+se\s+(?:₹|rs\.?|rupees?|[0-9])",

        # Havells from 500
        r"^(.+?)\s+from\s+(?:₹|rs\.?|rupees?|[0-9])",

        # Havells se purchase
        r"^(.+?)\s+se\s+(?:purchase|purchased|payment|paid|return|returned|credit)\b",

        # Havells ko 500
        r"^(.+?)\s+ko\s+(?:₹|rs\.?|rupees?|[0-9])",

        # Havells ko payment
        r"^(.+?)\s+ko\s+(?:payment|paid|pay)\b",

        # purchase 2000 from Havells
        r"(?:purchase|purchased|buy|bought|payment|paid|pay|return|returned|credit(?:\s+note)?)\b.*?\bfrom\s+(.+?)(?:\s+(?:for|of)\b|$)",

        # purchase from Havells
        r"(?:purchase|purchased|buy|bought|payment|paid|pay|return|returned|credit(?:\s+note)?)\b.*?\bfrom\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            value,
            re.IGNORECASE,
        )

        if not match:
            continue

        supplier = match.group(1).strip(" ,.-")

        supplier = re.sub(
            r"^(from|to|supplier)\s+",
            "",
            supplier,
            flags=re.IGNORECASE,
        ).strip(" ,.-")

        if supplier.lower() in {
            "purchase",
            "purchased",
            "buy",
            "bought",
            "payment",
            "paid",
            "pay",
            "return",
            "returned",
            "credit",
            "credit note",
        }:
            continue

        if supplier:
            return supplier

    return None


# ============================================================
# LOCAL EXTRACTION
# ============================================================

def local_extract_transaction(text: str):

    amount = extract_amount(text)
    transaction_type = extract_transaction_type(text)
    supplier_name = extract_supplier_name(text)

    print(
        "LOCAL EXTRACT:",
        {
            "transaction_type": transaction_type,
            "supplier_name": supplier_name,
            "amount": amount,
        },
        flush=True,
    )

    # We can safely return a transaction even when supplier
    # or amount is missing. Processor will ask for information.
    if transaction_type is None:
        return None

    return Transaction(
        transaction_type=transaction_type,
        supplier_name=supplier_name,
        amount=amount,
        payment_status=None,
        transaction_date=None,
        reference_number=None,
        notes=None,
    )


# ============================================================
# AI EXTRACTION
# ============================================================

def ai_extract_transaction(text: str) -> Transaction:

    if client is None:
        raise ValueError(
            "AI extraction service is not configured"
        )

    prompt = f"""
Extract transaction information from this user message.

Allowed transaction types:

PURCHASE
PAYMENT
RETURN
CREDIT_NOTE

Return ONLY valid JSON.

Required JSON structure:

{{
    "transaction_type": null,
    "supplier_name": null,
    "amount": null,
    "payment_status": null,
    "transaction_date": null,
    "reference_number": null,
    "notes": null
}}

Rules:

1. Understand English and Roman Hinglish only.
2. Do not require Devanagari Hindi.
3. Extract supplier name if explicitly mentioned.
4. Extract amount if explicitly mentioned.
5. Amount must be greater than zero.
6. If amount is explicitly negative, return amount null.
7. Extract date ONLY if explicitly mentioned.
8. Convert dates to YYYY-MM-DD.
9. Never assume today's date.
10. Never invent missing information.
11. Use null for missing fields.
12. Do not calculate balances.
13. Do not modify any database.

Examples:

"Havells se 500 ka maal liya"
=> PURCHASE, Havells, 500

"Havells se 500 ka payment kiya"
=> PAYMENT, Havells, 500

"Havells ka 500 ka maal return kiya"
=> RETURN, Havells, 500

"Havells ka 500 ka credit note"
=> CREDIT_NOTE, Havells, 500

User message:

{text}
"""

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            max_tokens=180,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={
                "type": "json_object"
            },
        )

    except Exception as e:
        print(
            "AI TRANSACTION EXTRACTION ERROR:",
            repr(e),
            flush=True,
        )

        raise ValueError(
            "AI extraction is temporarily unavailable. "
            "Please use a clear English/Roman Hinglish voice command "
            "with supplier, amount and transaction type."
        )

    if not response.choices:
        raise ValueError(
            "AI extractor returned no choices"
        )

    data = response.choices[0].message.content

    if not data:
        raise ValueError(
            "AI extractor returned empty response"
        )

    print(
        "AI TRANSACTION RAW:",
        repr(data),
        flush=True,
    )

    try:
        transaction = Transaction.model_validate_json(data)

    except Exception as e:
        print(
            "AI TRANSACTION VALIDATION ERROR:",
            repr(e),
            flush=True,
        )

        raise ValueError(
            "AI extractor returned invalid transaction data."
        )

    return transaction


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_transaction(text: str):

    if not isinstance(text, str):
        raise ValueError(
            "Transaction text must be text"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Transaction text cannot be empty"
        )

    # --------------------------------------------------------
    # FIRST: deterministic local extraction
    # --------------------------------------------------------

    transaction = local_extract_transaction(text)

    if transaction is not None:

        print(
            "LOCAL TRANSACTION EXTRACTION USED",
            flush=True,
        )

        transaction.transaction_date = None

        if transaction.amount is not None:
            if transaction.amount <= 0:
                transaction.amount = None

        return transaction

    # --------------------------------------------------------
    # SECOND: AI extraction
    # --------------------------------------------------------

    print(
        "FALLING BACK TO AI TRANSACTION EXTRACTION",
        flush=True,
    )

    transaction = ai_extract_transaction(text)

    if not has_explicit_date(text):
        transaction.transaction_date = None

    if transaction.amount is not None:
        if transaction.amount <= 0:
            transaction.amount = None

    return transaction