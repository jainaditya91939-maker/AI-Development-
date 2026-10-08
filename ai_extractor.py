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
# ROMAN HINGLISH NUMBER PARSER
# ============================================================

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,

    "ek": 1,
    "do": 2,
    "teen": 3,
    "tin": 3,
    "char": 4,
    "chaar": 4,
    "paanch": 5,
    "panch": 5,
    "che": 6,
    "chhe": 6,
    "saat": 7,
    "aath": 8,
    "nau": 9,
    "das": 10,

    "gyarah": 11,
    "barah": 12,
    "terah": 13,
    "chaudah": 14,
    "pandrah": 15,
    "solah": 16,
    "satrah": 17,
    "atharah": 18,
    "unnis": 19,
    "bees": 20,

    "ikkees": 21,
    "baees": 22,
    "teis": 23,
    "chaubees": 24,
    "pachis": 25,
    "chabbis": 26,
    "sattais": 27,
    "athais": 28,
    "untis": 29,
    "tees": 30,

    "ikattis": 31,
    "battis": 32,
    "taintees": 33,
    "chauntees": 34,
    "paintis": 35,
    "chattis": 36,
    "saintees": 37,
    "adtees": 38,
    "untalis": 39,
    "chalis": 40,

    "iktalis": 41,
    "bayalis": 42,
    "taitalis": 43,
    "chavalis": 44,
    "paintalis": 45,
    "chiyalis": 46,
    "saitalis": 47,
    "artalis": 48,
    "unchaas": 49,
    "pachaas": 50,

    "saath": 60,
    "sattar": 70,
    "assi": 80,
    "nabbe": 90,

    "hundred": 100,
    "hundred": 100,
    "sau": 100,
    "soo": 100,

    "thousand": 1000,
    "hazaar": 1000,
    "hazar": 1000,

    "lakh": 100000,
    "lac": 100000,

    "million": 1000000,
}


def parse_word_number(text: str):

    if not text:
        return None

    value = text.lower().strip()

    value = re.sub(
        r"[,\-]+",
        " ",
        value
    )

    words = value.split()

    if not words:
        return None

    total = 0
    current = 0
    found_number = False

    for word in words:

        if word in NUMBER_WORDS:

            number = NUMBER_WORDS[word]
            found_number = True

            if number in (100, 1000, 100000, 1000000):

                if current == 0:
                    current = 1

                current *= number

                if number >= 1000:
                    total += current
                    current = 0

            else:

                current += number

        else:

            # Ignore common amount words
            if word in {
                "rupee",
                "rupees",
                "rupay",
                "rupaye",
                "rs",
                "rs.",
                "ka",
                "ki",
                "ke",
                "only",
                "and",
                "aur",
            }:
                continue

            # Unknown word means this probably isn't
            # a pure number phrase.
            return None

    if not found_number:
        return None

    result = total + current

    if result > 0:
        return result

    return None


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amount(text: str):

    if not text:
        return None

    # --------------------------------------------------------
    # 1. Numeric amounts
    # --------------------------------------------------------

    patterns = [

        # ₹500
        r"₹\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",

        # rs 500 / rs. 500
        r"(?:rs\.?)\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",

        # 500 rupees / 500 rupee
        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*rupees?",

        # 500 rupay / 500 rupaye / 500 rupia
        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:rupay|rupaye|rupiya|rupiah)",

        # 500 ka / 500 ki / 500 ke
        r"(?:₹\s*)?([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:ka|ki|ke)(?=\s|$|[.,!?])",

        # 500
        # Only use standalone numbers when they are close to
        # transaction words, avoiding dates.
        r"\b([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\b",
    ]

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text,
            re.IGNORECASE
        )

        for match in matches:

            value = match.group(1).replace(
                ",",
                ""
            )

            try:

                amount = float(value)

                # Ignore obviously date-like values
                if amount >= 100000000:
                    continue

                if amount > 0:
                    return amount

            except ValueError:
                continue

    # --------------------------------------------------------
    # 2. Roman Hinglish word amounts
    # --------------------------------------------------------

    normalized_text = text.lower()

    # Look for phrases around common amount words.
    word_patterns = [

        r"\b([a-z]+(?:\s+[a-z]+){0,5})\s+(?:rupee|rupees|rupay|rupaye|rupiya|rupiah)\b",

        r"\b([a-z]+(?:\s+[a-z]+){0,5})\s+(?:ka|ki|ke)\b",
    ]

    for pattern in word_patterns:

        matches = re.finditer(
            pattern,
            normalized_text,
            re.IGNORECASE
        )

        for match in matches:

            phrase = match.group(1).strip()

            # Don't accidentally parse supplier name.
            result = parse_word_number(phrase)

            if result is not None and result > 0:
                return float(result)

    # --------------------------------------------------------
    # 3. Specific common Hinglish amount phrases
    # --------------------------------------------------------

    common_amounts = {

        "ek sau": 100,
        "do sau": 200,
        "teen sau": 300,
        "teen soo": 300,
        "tin sau": 300,
        "tin soo": 300,

        "char sau": 400,
        "chaar sau": 400,

        "paanch sau": 500,
        "panch sau": 500,

        "che sau": 600,
        "chhe sau": 600,

        "saat sau": 700,

        "aath sau": 800,

        "nau sau": 900,

        "ek hazaar": 1000,
        "ek hajar": 1000,
        "do hazaar": 2000,
        "do hajar": 2000,
        "teen hazaar": 3000,
        "teen hajar": 3000,
        "chaar hazaar": 4000,
        "paanch hazaar": 5000,
        "das hazaar": 10000,

        "bees hazaar": 20000,
        "pachaas hazaar": 50000,
        "ek lakh": 100000,
    }

    for phrase, amount in common_amounts.items():

        if re.search(
            r"\b"
            + re.escape(phrase)
            + r"\b",
            normalized_text
        ):
            return float(amount)

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
        r"^(.+?)\s+(?:₹|rs\.?|rupees?|rupay|rupaye)\s*[0-9][0-9,]*(?:\.[0-9]+)?\s+(?:purchase|payment|return|credit)\b",

        # Havells se 500
        r"^(.+?)\s+se\s+(?:₹|rs\.?|rupees?|rupay|rupaye|[0-9])",

        # Havells from 500
        r"^(.+?)\s+from\s+(?:₹|rs\.?|rupees?|rupay|rupaye|[0-9])",

        # Havells se purchase
        r"^(.+?)\s+se\s+(?:purchase|purchased|payment|paid|return|returned|credit)\b",

        # Havells ko 500
        r"^(.+?)\s+ko\s+(?:₹|rs\.?|rupees?|rupay|rupaye|[0-9])",

        # Havells ko payment
        r"^(.+?)\s+ko\s+(?:payment|paid|pay)\b",

        # purchase from Havells
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

        supplier = match.group(1).strip(
            " ,.-"
        )

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

    transaction_type = extract_transaction_type(
        text
    )

    supplier_name = extract_supplier_name(
        text
    )

    print(
        "LOCAL EXTRACT:",
        {
            "transaction_type": transaction_type,
            "supplier_name": supplier_name,
            "amount": amount,
        },
        flush=True,
    )

    # We can safely return a transaction even when
    # supplier or amount is missing.
    # Processor will ask for information.
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
5. Understand numeric amounts such as 500, 1000, 2500.
6. Understand Roman Hinglish amounts such as:
   - teen sau = 300
   - teen soo = 300
   - paanch sau = 500
   - paanch sau = 500
   - do hazaar = 2000
   - ek hazaar = 1000
   - ek lakh = 100000
7. Amount must be greater than zero.
8. If amount is explicitly negative, return amount null.
9. Extract date ONLY if explicitly mentioned.
10. Convert dates to YYYY-MM-DD.
11. Never assume today's date.
12. Never invent missing information.
13. Use null for missing fields.
14. Do not calculate balances.
15. Do not modify any database.

Examples:

"Havells se 500 ka maal liya"
=> PURCHASE, Havells, 500

"Havells se 500 rupay ka payment kiya"
=> PAYMENT, Havells, 500

"Havells se teen sau rupay ka purchase kiya"
=> PURCHASE, Havells, 300

"Havells se teen soo rupay ka purchase kiya"
=> PURCHASE, Havells, 300

"Havells se paanch sau rupaye ka maal liya"
=> PURCHASE, Havells, 500

"Havells se do hazaar rupay ka purchase kiya"
=> PURCHASE, Havells, 2000

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

        transaction = Transaction.model_validate_json(
            data
        )

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

    transaction = local_extract_transaction(
        text
    )

    if transaction is not None:

        print(
            "LOCAL TRANSACTION EXTRACTION USED",
            flush=True,
        )

        # Date should only be accepted if explicitly mentioned.
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

    transaction = ai_extract_transaction(
        text
    )

    if not has_explicit_date(text):

        transaction.transaction_date = None

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    return transaction