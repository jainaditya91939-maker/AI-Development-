import os
import re
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from schemas import Transaction


# ============================================================
# ENV / OPENROUTER
# ============================================================

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
# ROMAN-HINGLISH NUMBER WORDS
# ============================================================

NUMBER_WORDS = {

    "zero": 0,

    "ek": 1,
    "one": 1,

    "do": 2,
    "two": 2,

    "teen": 3,
    "three": 3,

    "char": 4,
    "chaar": 4,
    "four": 4,

    "paanch": 5,
    "panch": 5,
    "five": 5,

    "che": 6,
    "chhe": 6,
    "chhah": 6,
    "six": 6,

    "saat": 7,
    "seven": 7,

    "aath": 8,
    "eight": 8,

    "nau": 9,
    "no": 9,
    "nine": 9,

    "dus": 10,
    "das": 10,
    "ten": 10,

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
    "tees": 30,
    "chalees": 40,
    "chaalis": 40,
    "pachaas": 50,
    "saath": 60,
    "sattar": 70,
    "assi": 80,
    "nabbe": 90,
}


# Common speech-recognition forms
KNOWN_AMOUNTS = {

    "paanso": 500,
    "paanchso": 500,
    "panchso": 500,

    "hazaar": 1000,
    "hazar": 1000,
    "thousand": 1000,

    "do hazaar": 2000,
    "dohazar": 2000,

    "teen hazaar": 3000,

    "chaar hazaar": 4000,

    "paanch hazaar": 5000,
    "panch hazaar": 5000,

    "das hazaar": 10000,
    "dus hazaar": 10000,

    "bees hazaar": 20000,

    "lakh": 100000,
    "lac": 100000,
}


def parse_hinglish_number(value: str):

    value = value.lower().strip()

    value = re.sub(
        r"[^a-z\s-]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    if not value:
        return None

    compact = (
        value
        .replace(" ", "")
        .replace("-", "")
    )

    if compact in KNOWN_AMOUNTS:

        return float(
            KNOWN_AMOUNTS[compact]
        )

    if value in KNOWN_AMOUNTS:

        return float(
            KNOWN_AMOUNTS[value]
        )

    tokens = (
        value
        .replace("-", " ")
        .split()
    )

    total = 0
    current = 0
    found = False

    for token in tokens:

        if token in NUMBER_WORDS:

            current += NUMBER_WORDS[token]

            found = True

        elif token in {
            "sau",
            "hundred",
        }:

            if current == 0:
                current = 1

            total += current * 100

            current = 0

            found = True

        elif token in {
            "hazaar",
            "hazar",
            "thousand",
        }:

            if current == 0:
                current = 1

            total += current * 1000

            current = 0

            found = True

        elif token in {
            "lakh",
            "lac",
        }:

            if current == 0:
                current = 1

            total += current * 100000

            current = 0

            found = True

        else:

            return None

    if not found:
        return None

    result = total + current

    if result > 0:
        return float(result)

    return None


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amount(text: str):

    # ----------------------------------------
    # NORMAL NUMERIC AMOUNTS
    # ----------------------------------------

    numeric_patterns = [

        r"₹\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",

        r"(?:rs\.?|rupees?)\s*"
        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",

        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)"
        r"\s*rupees?",

        r"(?:₹\s*)?"
        r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)"
        r"\s*(?:ka|ki|ke)\b",
    ]

    for pattern in numeric_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if not match:
            continue

        try:

            amount = float(
                match.group(1)
                .replace(",", "")
            )

            if amount > 0:
                return amount

        except ValueError:

            pass


    # ----------------------------------------
    # ROMAN-HINGLISH SPOKEN AMOUNTS
    # ----------------------------------------

    value = text.lower().strip()

    value = re.sub(
        r"[,.!?]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


    # ----------------------------------------
    # EXACT COMMON AMOUNTS
    # ----------------------------------------

    for phrase, amount in sorted(
        KNOWN_AMOUNTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):

        if re.search(
            rf"\b{re.escape(phrase)}\b",
            value,
            re.IGNORECASE,
        ):

            return float(amount)


    # ----------------------------------------
    # "paanch sau rupee"
    # "do hazaar rupees"
    # "paanch hazaar ka"
    # ----------------------------------------

    word_pattern = (

        r"\b("

        r"(?:"
        r"zero|ek|one|do|two|teen|three|"
        r"char|chaar|four|paanch|panch|five|"
        r"che|chhe|chhah|six|saat|seven|"
        r"aath|eight|nau|no|nine|dus|das|"
        r"ten|gyarah|barah|terah|chaudah|"
        r"pandrah|solah|satrah|atharah|"
        r"unnis|bees|tees|chalees|chaalis|"
        r"pachaas|saath|sattar|assi|nabbe|"
        r"sau|hundred|hazaar|hazar|thousand|"
        r"lakh|lac"
        r")"

        r"(?:[- ]+"

        r"(?:"
        r"zero|ek|one|do|two|teen|three|"
        r"char|chaar|four|paanch|panch|five|"
        r"che|chhe|chhah|six|saat|seven|"
        r"aath|eight|nau|no|nine|dus|das|"
        r"ten|gyarah|barah|terah|chaudah|"
        r"pandrah|solah|satrah|atharah|"
        r"unnis|bees|tees|chalees|chaalis|"
        r"pachaas|saath|sattar|assi|nabbe|"
        r"sau|hundred|hazaar|hazar|thousand|"
        r"lakh|lac"
        r")"

        r")*"

        r")\s+"

        r"(?:"
        r"rupees?|rupaye|rupay|rs\.?|ka|ki|ke"
        r")\b"
    )


    match = re.search(
        word_pattern,
        value,
        re.IGNORECASE,
    )

    if match:

        amount = parse_hinglish_number(
            match.group(1)
        )

        if amount is not None:
            return amount


    # ----------------------------------------
    # SPOKEN AMOUNT WITHOUT "RUPEES"
    # Example:
    # "paanch sau ka purchase"
    # ----------------------------------------

    words = re.findall(
        r"\b[a-z]+(?:[- ][a-z]+){0,4}\b",
        value,
    )

    for phrase in words:

        amount = parse_hinglish_number(
            phrase
        )

        if (
            amount is not None
            and amount >= 100
        ):

            return amount


    return None


# ============================================================
# TRANSACTION TYPE
# ============================================================

def extract_transaction_type(text: str):

    value = text.lower().strip()


    # ----------------------------------------
    # CREDIT NOTE
    # ----------------------------------------

    credit_words = [

        "credit note",

        "credit_note",

        "credit",

    ]

    for word in credit_words:

        if word in value:

            return "CREDIT_NOTE"


    # ----------------------------------------
    # RETURN
    # ----------------------------------------

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


    # ----------------------------------------
    # PAYMENT
    # ----------------------------------------

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


    # ----------------------------------------
    # PURCHASE
    # ----------------------------------------

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


    # ----------------------------------------
    # IMPORTANT:
    #
    # Check supplier + "se/from/to/ko"
    # BEFORE transaction-at-start patterns.
    #
    # Example:
    # Haldiram se paanso rupee ka purchase kiya
    #
    # Supplier must be:
    # Haldiram
    # ----------------------------------------

    relation_match = re.search(
        r"^(.+?)\s+(?:se|from|to|ko)\b",
        value,
        re.IGNORECASE,
    )


    if relation_match:

        supplier = (
            relation_match
            .group(1)
            .strip(" ,.-")
        )

        relation_prefix = supplier.lower()


        # Do not treat:
        # "purchase 2000"
        # as supplier.
        transaction_prefix_words = (

            "purchase ",

            "purchased ",

            "buy ",

            "bought ",

            "payment ",

            "paid ",

            "pay ",

            "return ",

            "returned ",

            "credit ",

        )


        if (
            supplier
            and not relation_prefix.startswith(
                transaction_prefix_words
            )
        ):

            return supplier


    # ----------------------------------------
    # OTHER SUPPLIER PATTERNS
    # ----------------------------------------

    patterns = [

        # purchase/payment/return FROM supplier
        r"(?:purchase|purchased|buy|bought|"
        r"payment|paid|pay|return|returned|"
        r"credit(?:\s+note)?)\b"
        r".*?\bfrom\s+"
        r"(.+?)(?:\s+(?:for|of)\b|$)",


        # Supplier ka ... return/payment/purchase
        r"^(.+?)\s+ka\b.*?\b"
        r"(?:purchase|purchased|buy|bought|"
        r"payment|paid|pay|return|returned|"
        r"credit(?:\s+note)?)\b",


        # Supplier purchase
        r"^(.+?)\s+"
        r"(?:purchase|purchased|buy|bought)\b",


        # Supplier payment
        r"^(.+?)\s+"
        r"(?:payment|paid|pay)\b",


        # Supplier return
        r"^(.+?)\s+"
        r"(?:return|returned)\b",


        # Supplier credit note
        r"^(.+?)\s+credit\s+note\b",


        # Supplier 500 purchase
        r"^(.+?)\s+"
        r"(?:₹|rs\.?|rupees?)\s*"
        r"[0-9][0-9,]*(?:\.[0-9]+)?\s+"
        r"(?:purchase|payment|return|credit)\b",

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            value,
            re.IGNORECASE,
        )

        if not match:
            continue


        supplier = (
            match
            .group(1)
            .strip(" ,.-")
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

    transaction_type = (
        extract_transaction_type(text)
    )

    supplier_name = (
        extract_supplier_name(text)
    )


    print(
        "LOCAL EXTRACT:",
        {
            "transaction_type":
                transaction_type,

            "supplier_name":
                supplier_name,

            "amount":
                amount,
        },
        flush=True,
    )


    # If transaction type is unknown,
    # use AI fallback.
    #
    # If supplier or amount is missing,
    # still return the transaction.
    # processor.py will ask for missing
    # information.

    if transaction_type is None:

        return None


    return Transaction(

        transaction_type=
            transaction_type,

        supplier_name=
            supplier_name,

        amount=
            amount,

        payment_status=
            None,

        transaction_date=
            None,

        reference_number=
            None,

        notes=
            None,
    )


# ============================================================
# AI EXTRACTION FALLBACK
# ============================================================

def ai_extract_transaction(text: str):

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

6. If amount is explicitly negative,
   return amount null.

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

        response = (
            client.chat.completions.create(

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
        )


    except Exception as e:

        print(
            "AI TRANSACTION EXTRACTION ERROR:",
            repr(e),
            flush=True,
        )


        raise ValueError(

            "AI extraction is temporarily unavailable. "

            "Please use a clear English/Roman Hinglish "
            "voice command with supplier, amount "
            "and transaction type."

        )


    if not response.choices:

        raise ValueError(
            "AI extractor returned no choices"
        )


    data = (
        response
        .choices[0]
        .message
        .content
    )


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

        transaction = (
            Transaction
            .model_validate_json(data)
        )


    except Exception as e:

        print(
            "AI TRANSACTION VALIDATION ERROR:",
            repr(e),
            flush=True,
        )


        raise ValueError(
            "AI extractor returned "
            "invalid transaction data."
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


    # ========================================================
    # FIRST:
    # DETERMINISTIC LOCAL EXTRACTION
    # ========================================================

    transaction = (
        local_extract_transaction(text)
    )


    if transaction is not None:

        print(
            "LOCAL TRANSACTION EXTRACTION USED",
            flush=True,
        )


        # We only use an explicit date
        # when one is actually provided.
        transaction.transaction_date = None


        if transaction.amount is not None:

            if transaction.amount <= 0:

                transaction.amount = None


        return transaction


    # ========================================================
    # SECOND:
    # AI FALLBACK
    #
    # Only used when local extraction cannot
    # identify the transaction type.
    # ========================================================

    print(
        "FALLING BACK TO AI TRANSACTION EXTRACTION",
        flush=True,
    )


    transaction = (
        ai_extract_transaction(text)
    )


    if not has_explicit_date(text):

        transaction.transaction_date = None


    if transaction.amount is not None:

        if transaction.amount <= 0:

            transaction.amount = None


    return transaction