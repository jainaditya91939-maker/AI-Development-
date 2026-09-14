import os
import re
from datetime import date

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
# DATE CHECK
# ==========================================

def has_explicit_date(text: str) -> bool:
    date_pattern = re.compile(
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

            |

            \b\d{1,2}\s+
            (?:
                जनवरी|
                फरवरी|
                फ़रवरी|
                मार्च|
                अप्रैल|
                मई|
                जून|
                जुलाई|
                अगस्त|
                सितंबर|
                सितम्बर|
                अक्टूबर|
                नवंबर|
                नवम्बर|
                दिसंबर|
                दिसम्बर
            )
            \s+\d{4}\b
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    return bool(date_pattern.search(text))


# ==========================================
# AMOUNT EXTRACTION
# ==========================================

def extract_amount(text: str):
    """
    Extract common Indian currency formats.

    Examples:
    ₹2000
    ₹ 2000
    2000 rupees
    2000 रुपये
    2,000
    """

    patterns = [
        r'₹\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)',
        r'rs\.?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)',
        r'rupees?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)',
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*rupees?',
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*रुपये',
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*रुपए',
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*रुपया',
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            value = match.group(1).replace(",", "")

            try:
                amount = float(value)

                if amount > 0:
                    return amount

            except ValueError:
                pass

    return None


# ==========================================
# TRANSACTION TYPE EXTRACTION
# ==========================================

def extract_transaction_type(text: str):
    value = text.lower().strip()

    # PURCHASE
    purchase_words = [
        "purchase",
        "purchased",
        "buy",
        "bought",
        "purchase hua",
        "purchase ki",
        "purchase kar",
        "खरीद",
        "खरीदा",
        "खरीदी",
        "खरीद हुआ",
        "परचेस",
        "परचेज",
    ]

    for word in purchase_words:
        if word in value:
            return "PURCHASE"

    # PAYMENT
    payment_words = [
        "payment",
        "paid",
        "pay",
        "payment ki",
        "payment hua",
        "दे दिया",
        "दिया",
        "भुगतान",
        "पेमेंट",
    ]

    for word in payment_words:
        if word in value:
            return "PAYMENT"

    # RETURN
    return_words = [
        "return",
        "returned",
        "return hua",
        "return ki",
        "वापस",
        "वापसी",
        "रिटर्न",
    ]

    for word in return_words:
        if word in value:
            return "RETURN"

    # CREDIT NOTE
    credit_words = [
        "credit note",
        "credit_note",
        "credit",
        "क्रेडिट नोट",
        "क्रेडिट",
    ]

    for word in credit_words:
        if word in value:
            return "CREDIT_NOTE"

    return None


# ==========================================
# SUPPLIER EXTRACTION
# ==========================================

def extract_supplier_name(text: str):
    """
    Extract supplier name from common English,
    Hindi and Hinglish voice patterns.

    Examples:
    ABC Electricals se ₹2000 ka purchase hua
    ABC Electricals ko ₹500 payment ki
    ABC Electricals se purchase hua
    """

    patterns = [
        # "... se ..."
        r'(.+?)\s+se\s+(?:₹|rs\.?|rupees?|[0-9])',
        r'(.+?)\s+se\s+(?:purchase|payment|return|credit)',
        r'(.+?)\s+se\s+.*?(?:hua|hui|ki|kiya|kiye)',

        # "... ko ..."
        r'(.+?)\s+ko\s+(?:₹|rs\.?|rupees?|[0-9])',
        r'(.+?)\s+ko\s+(?:payment|pay|paid)',

        # Hindi patterns
        r'(.+?)\s+से\s+(?:₹|[0-9])',
        r'(.+?)\s+से\s+.*?(?:परचेस|परचेज|पेमेंट|रिटर्न)',

        r'(.+?)\s+को\s+(?:₹|[0-9])',
        r'(.+?)\s+को\s+.*?(?:पेमेंट|भुगतान)',
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            supplier = match.group(1).strip()

            supplier = re.sub(
                r'^(from|to|supplier)\s+',
                '',
                supplier,
                flags=re.IGNORECASE
            )

            supplier = supplier.strip(" ,.-")

            if supplier:
                return supplier

    return None


# ==========================================
# LOCAL EXTRACTION
# ==========================================

def local_extract_transaction(text: str):
    """
    Try deterministic extraction first.

    This avoids OpenRouter completely for normal
    voice transaction commands.
    """

    amount = extract_amount(text)
    transaction_type = extract_transaction_type(text)
    supplier_name = extract_supplier_name(text)

    # We only trust local extraction when we have
    # the minimum important information.
    if (
        transaction_type is None
        or supplier_name is None
        or amount is None
    ):
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


# ==========================================
# AI FALLBACK
# ==========================================

def ai_extract_transaction(text: str) -> Transaction:

    prompt = f"""
Extract transaction information from this user message.

Allowed types:
PURCHASE, PAYMENT, RETURN, CREDIT_NOTE

Return ONLY JSON with:

transaction_type
supplier_name
amount
payment_status
transaction_date
reference_number
notes

Rules:

1. Understand English, Hindi and Hinglish.
2. Extract supplier name if explicitly mentioned.
3. Extract amount if explicitly mentioned.
4. Amount must be greater than zero.
5. If amount is explicitly negative, return amount null.
6. Extract date ONLY if explicitly mentioned.
7. Convert dates to YYYY-MM-DD.
8. Never assume today's date.
9. Never invent missing information.
10. Use null for missing values.
11. Do not calculate balances.
12. Do not modify any database.
13. Do not create, update or delete transactions.

Examples:

"12 September 2026" -> "2026-09-12"
"12/09/2026" -> "2026-09-12"
"12 सितंबर 2026" -> "2026-09-12"

User message:
{text}
"""

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

    data = response.choices[0].message.content

    if not data:
        raise ValueError(
            "AI extractor returned empty response"
        )

    transaction = Transaction.model_validate_json(data)

    return transaction


# ==========================================
# MAIN EXTRACTION FUNCTION
# ==========================================

def extract_transaction(text: str) -> Transaction:

    text = text.strip()

    if not text:
        raise ValueError(
            "Transaction text cannot be empty"
        )

    # ==========================================
    # STEP 1: LOCAL EXTRACTION
    # ==========================================

    transaction = local_extract_transaction(text)

    if transaction is not None:

        # Hard date safety:
        # Local extractor never invents dates.
        transaction.transaction_date = None

        # Hard amount safety
        if transaction.amount is not None:
            if transaction.amount <= 0:
                transaction.amount = None

        return transaction

    # ==========================================
    # STEP 2: AI FALLBACK
    # ==========================================

    transaction = ai_extract_transaction(text)

    # ==========================================
    # HARD DATE SAFETY CHECK
    # ==========================================

    if not has_explicit_date(text):
        transaction.transaction_date = None

    # ==========================================
    # HARD AMOUNT SAFETY CHECK
    # ==========================================

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    return transaction