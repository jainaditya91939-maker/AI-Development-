import os
import re
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from schemas import Transaction


load_dotenv()


# ============================================================
# OPENROUTER CLIENT
# ============================================================

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# DATE DETECTION
# ============================================================

def has_explicit_date(text: str) -> bool:

    date_pattern = re.compile(
        r"""
        (
            \\b\\d{1,2}[\\/\\-.]\\d{1,2}[\\/\\-.]\\d{4}\\b

            |

            \\b\\d{1,2}\\s+
            (?:
                january|february|march|april|may|june|
                july|august|september|october|november|december|
                jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec
            )
            \\s+\\d{4}\\b

            |

            \\b
            (?:
                january|february|march|april|may|june|
                july|august|september|october|november|december|
                jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec
            )
            \\s+\\d{1,2},?\\s+\\d{4}\\b
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    return bool(date_pattern.search(text))


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amount(text: str):

    """
    Generic English / Roman-Hinglish amount extraction.

    Supports:
        ₹500
        ₹ 500
        Rs 500
        Rs. 500
        500 rupees
        500 rs
        500 ka
        500 ki
        500 ke
    """

    patterns = [

        r'₹\\s*([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)',
        r'\\brs\\.?\\s*([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)',
        r'\\brupees?\\s*([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)',
        r'([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)\\s*rupees?',
        r'([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)\\s*rs\\.?',

        r'(?:₹\\s*)?([0-9]+(?:,[0-9]+)*(?:\\.[0-9]+)?)\\s*(?:ka|ki|ke)(?=\\s|$|[.,!?])',
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

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

    credit_words = [
        "credit note",
        "credit_note",
        "credit",
    ]

    for word in credit_words:
        if word in value:
            return "CREDIT_NOTE"

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

    payment_words = [
        "payment",
        "paid",
        "pay",
        "payment ki",
        "payment hua",
        "payment kiya",
        "pay kiya",
    ]

    for word in payment_words:
        if word in value:
            return "PAYMENT"

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
        "samaan kharida",
    ]

    for word in purchase_words:
        if word in value:
            return "PURCHASE"

    return None


# ============================================================
# SUPPLIER NAME
# ============================================================

def extract_supplier_name(text: str):

    patterns = [
        r'(.+?)\\s+se\\s+(?:₹|rs\\.?|rupees?|[0-9])',
        r'(.+?)\\s+se\\s+(?:purchase|purchased|payment|return|credit)',
        r'(.+?)\\s+se\\s+.*?(?:hua|hui|ki|kiya|kiye|liya|liye)',
        r'(.+?)\\s+ko\\s+(?:₹|rs\\.?|rupees?|[0-9])',
        r'(.+?)\\s+ko\\s+(?:payment|pay|paid)',
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if not match:
            continue

        supplier = match.group(1).strip()

        supplier = re.sub(
            r'^(from|to|supplier)\\s+',
            '',
            supplier,
            flags=re.IGNORECASE
        )

        supplier = supplier.strip(" ,.-")

        if supplier:
            return supplier

    return None


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
            "amount": amount
        },
        flush=True
    )

    # --------------------------------------------------------
    # Transaction schema requires transaction_type.
    #
    # Never create:
    #
    # transaction_type=None
    #
    # If type is missing, AI extraction handles it.
    # --------------------------------------------------------

    if transaction_type is None:

        print(
            "LOCAL EXTRACTION INCOMPLETE: transaction type not found",
            flush=True
        )

        return None

    return Transaction(
        transaction_type=transaction_type,
        supplier_name=supplier_name,
        amount=amount,
        payment_status=None,
        transaction_date=None,
        reference_number=None,
        notes=None
    )


# ============================================================
# AI EXTRACTION
# ============================================================

def ai_extract_transaction(text: str) -> Transaction:

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
    "transaction_type": "PURCHASE",
    "supplier_name": null,
    "amount": null,
    "payment_status": null,
    "transaction_date": null,
    "reference_number": null,
    "notes": null
}}

Rules:

1. Understand English and Roman Hinglish only.

2. Extract supplier name if explicitly mentioned.

3. Extract amount if explicitly mentioned.

4. Amount must be greater than zero.

5. If amount is explicitly negative, return amount null.

6. Extract date ONLY if explicitly mentioned.

7. Convert dates to YYYY-MM-DD.

8. Never assume today's date.

9. Never invent missing information.

10. Use null for missing fields.

11. Do not calculate balances.

12. Do not modify any database.

13. Do not create, update or delete transactions.

14. Understand common speech-recognition mistakes.

15. Currency may be transcribed incorrectly.
    For example:
    "Rs 200"




    should all be interpreted as amount 200.

16. For a normal purchase such as:
    "Havells se 500 ka maal liya"
    use PURCHASE.

17. For:
    "Havells se 500 ka payment kiya"
    use PAYMENT.

18. For:
    "Havells ka 500 ka maal return kiya"
    use RETURN.

19. For:
    "Havells ka 500 ka credit note"
    use CREDIT_NOTE.

User message:

{text}
"""

    try:

        response = client.chat.completions.create(
            model="openrouter/free",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_object"
            }
        )

    except Exception as e:

        print(
            "AI TRANSACTION EXTRACTION ERROR:",
            repr(e),
            flush=True
        )

        raise ValueError(
            f"AI transaction extraction failed: {e}"
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
        flush=True
    )

    try:

        transaction = Transaction.model_validate_json(
            data
        )

    except Exception as e:

        print(
            "AI TRANSACTION VALIDATION ERROR:",
            repr(e),
            flush=True
        )

        raise ValueError(
            f"AI transaction data validation failed: {e}"
        )

    return transaction


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_transaction(text: str):

    text = text.strip()

    if not text:

        raise ValueError(
            "Transaction text cannot be empty"
        )

    # --------------------------------------------------------
    # FIRST: LOCAL EXTRACTION
    # --------------------------------------------------------

    transaction = local_extract_transaction(
        text
    )

    if transaction is not None:

        print(
            "LOCAL TRANSACTION EXTRACTION USED",
            flush=True
        )

        # Local extractor does not parse dates.
        transaction.transaction_date = None

        # Validate amount
        if transaction.amount is not None:

            if transaction.amount <= 0:
                transaction.amount = None

        return transaction

    # --------------------------------------------------------
    # SECOND: AI EXTRACTION
    # --------------------------------------------------------

    print(
        "FALLING BACK TO AI TRANSACTION EXTRACTION",
        flush=True
    )

    transaction = ai_extract_transaction(
        text
    )

    # --------------------------------------------------------
    # Never invent a date
    # --------------------------------------------------------

    if not has_explicit_date(text):

        transaction.transaction_date = None

    # --------------------------------------------------------
    # Amount validation
    # --------------------------------------------------------

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    return transaction