import os
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
# CHECK EXPLICIT DATE
# ==========================================

def has_explicit_date(text: str) -> bool:
    """
    Check whether the user explicitly mentioned
    a transaction date.
    """

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
                jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec
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
# TRANSACTION EXTRACTION
# ==========================================

def extract_transaction(text: str) -> Transaction:

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
9. Never invent missing information. Use null.
10. Do not calculate balances.
11. Do not modify any database.
12. Do not create, update or delete transactions.

Examples:
"12 September 2026" -> "2026-09-12"
"12/09/2026" -> "2026-09-12"
"12 सितंबर 2026" -> "2026-09-12"

User message:
{text}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",

        # OpenRouter currently has very low remaining credits.
        # 120 tokens is enough for this small JSON response.
        max_tokens=120,

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

    transaction = Transaction.model_validate_json(data)


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