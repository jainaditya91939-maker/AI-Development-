import os
import re
import requests
from difflib import SequenceMatcher


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


# ============================================================
# AUTHENTICATED API
# ============================================================

def _headers(token: str):
    if not token:
        raise ValueError("Authentication token is required")

    return {
        "Authorization": f"Bearer {token}"
    }


def get_suppliers(token: str):
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers",
        headers=_headers(token),
        timeout=20
    )

    response.raise_for_status()
    return response.json()


def get_supplier_summary(token: str):
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers/summary",
        headers=_headers(token),
        timeout=20
    )

    response.raise_for_status()
    return response.json()


def get_supplier_ledger(
    supplier_id: int,
    token: str
):
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers/{supplier_id}/ledger",
        headers=_headers(token),
        timeout=20
    )

    response.raise_for_status()
    return response.json()


# ============================================================
# CREATE SUPPLIER
# ============================================================

def create_supplier(
    name: str,
    token: str
):
    """
    Create a supplier inside the authenticated business.
    """

    if not name or not str(name).strip():
        raise ValueError("Supplier name is required")

    normalized_name = normalize_supplier_display_name(name)

    data = {
        "name": normalized_name,
        "phone": None,
        "address": None,
    }

    response = requests.post(
        f"{BACKEND_URL}/api/v1/suppliers",
        json=data,
        headers=_headers(token),
        timeout=20
    )

    if response.status_code == 409:
        return {
            "status": "EXISTS",
            "message": "Supplier already exists"
        }

    response.raise_for_status()

    return response.json()


# ============================================================
# SUPPLIER NAME NORMALIZATION
# ============================================================

def normalize_supplier_name(name: str) -> str:

    if not name:
        return ""

    value = str(name).lower().strip()

    # ========================================================
    # HINDI / HINGLISH SPEECH RECOGNITION ALIASES
    # ========================================================

    hindi_aliases = {

        # HAVELLS
        "हैवेल्स": "havells",
        "हैवेल": "havells",
        "हैवल्स": "havells",
        "हैवल": "havells",
        "हैवेलस": "havells",
        "हैवन्स": "havells",
        "हैवंत": "havells",

        # POLYCAB
        "पॉलीकैब": "polycab",
        "पॉली कैब": "polycab",
        "पोलिकैब": "polycab",
        "पोलीकैब": "polycab",
        "पोली कैब": "polycab",
        "होलीकैब": "polycab",
        "होली कैब": "polycab",
        "होलीकेब": "polycab",
        "होली केब": "polycab",
        "पॉली केब": "polycab",
        "पोलि कैब": "polycab",

        # ENGLISH ASR VARIANTS
        "policy": "polycab",
        "pollycab": "polycab",
        "poly cab": "polycab",
        "polly cab": "polycab",

        # ANCHOR
        "एंकर": "anchor",

        # FINOLEX
        "फिनोलेक्स": "finolex",

        # SCHNEIDER
        "श्नाइडर": "schneider",
        "स्नाइडर": "schneider",

        # PHILIPS
        "फिलिप्स": "philips",

        # CROMPTON
        "क्रॉम्पटन": "crompton",
        "क्रोम्पटन": "crompton",

        # LEGRAND
        "लेग्रैंड": "legrand",
        "लेग्रां": "legrand",

        # WIPRO
        "विप्रो": "wipro",

        # BAJAJ
        "बजाज": "bajaj",

        # RR
        "आरआर": "rr",
        "आर आर": "rr",

        # COMMON BUSINESS WORDS
        "इलेक्ट्रिकल": "electrical",
        "इलेक्ट्रिकल्स": "electricals",
        "ट्रेडर्स": "traders",
        "ट्रेडर": "trader",
        "एंटरप्राइजेज": "enterprises",
        "एंटरप्राइज": "enterprise",
        "इंडस्ट्रीज": "industries",
        "इंडस्ट्री": "industry",
        "कॉर्पोरेशन": "corporation",
        "कंपनी": "company",

        # COMMON ASR LETTERS
        "एबीसी": "abc",
        "ए बी सी": "abc",
        "आरएस": "rs",
        "आर एस": "rs",
        "एबी": "ab",
        "ए बी": "ab",
        "एक्सवाईजेड": "xyz",
        "एक्स वाई जेड": "xyz",
    }

    if value in hindi_aliases:
        return hindi_aliases[value]

    # ========================================================
    # PHRASE REPLACEMENTS
    # ========================================================

    phrase_aliases = {
        "एबीसी इलेक्ट्रिकल": "abc electrical",
        "ए बी सी इलेक्ट्रिकल": "abc electrical",
        "एबीसी इलेक्ट्रिकल्स": "abc electricals",
        "ए बी सी इलेक्ट्रिकल्स": "abc electricals",

        "पॉलीकैब इलेक्ट्रिकल": "polycab electrical",
        "पॉलीकैब इलेक्ट्रिकल्स": "polycab electricals",

        "हैवेल्स इलेक्ट्रिकल": "havells electrical",
        "हैवेल्स इलेक्ट्रिकल्स": "havells electricals",

        "एंकर इलेक्ट्रिकल": "anchor electrical",
        "फिनोलेक्स इलेक्ट्रिकल": "finolex electrical",

        "क्रॉम्पटन इलेक्ट्रिकल": "crompton electrical",
        "लेग्रैंड इलेक्ट्रिकल": "legrand electrical",
        "श्नाइडर इलेक्ट्रिकल": "schneider electrical",
    }

    if value in phrase_aliases:
        return phrase_aliases[value]

    # Longest phrases first
    for hindi_phrase, english_phrase in sorted(
        phrase_aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        value = value.replace(
            hindi_phrase,
            english_phrase
        )

    # ========================================================
    # WORD LEVEL REPLACEMENT
    # ========================================================

    words = value.split()
    converted_words = []

    for word in words:
        converted_words.append(
            hindi_aliases.get(word, word)
        )

    value = " ".join(converted_words)

    # ========================================================
    # REMOVE COMPANY SUFFIXES
    # ========================================================

    value = re.sub(
        r"\b(private limited|pvt ltd|pvt\. ltd\.|"
        r"private ltd|limited|ltd|llp|incorporated|inc)\b",
        " ",
        value
    )

    # ========================================================
    # REMOVE PUNCTUATION
    # ========================================================

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    # ========================================================
    # NORMALIZE WHITESPACE
    # ========================================================

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# DISPLAY NAME NORMALIZATION
# ============================================================

def normalize_supplier_display_name(name: str) -> str:

    if not name:
        return ""

    original = str(name).strip()

    normalized = normalize_supplier_name(original)

    if not normalized:
        return original

    words = normalized.split()

    final_words = []

    for word in words:

        # Preserve common acronyms
        if word in {
            "abc",
            "xyz",
            "rs",
            "rr",
            "ab",
            "hv",
            "lt",
            "ht"
        }:
            final_words.append(word.upper())
            continue

        final_words.append(
            word[:1].upper() + word[1:]
        )

    return " ".join(final_words)


# ============================================================
# TOKENIZATION
# ============================================================

def supplier_tokens(name: str):

    normalized = normalize_supplier_name(name)

    if not normalized:
        return set()

    return set(normalized.split())


# ============================================================
# SUPPLIER SIMILARITY
# ============================================================

def supplier_similarity(
    name1: str,
    name2: str
) -> float:

    a = normalize_supplier_name(name1)
    b = normalize_supplier_name(name2)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    a_tokens = supplier_tokens(name1)
    b_tokens = supplier_tokens(name2)

    if not a_tokens or not b_tokens:
        return 0.0

    # ========================================================
    # TOKEN OVERLAP
    # ========================================================

    common_tokens = a_tokens.intersection(
        b_tokens
    )

    if common_tokens:

        smaller_count = min(
            len(a_tokens),
            len(b_tokens)
        )

        containment_score = (
            len(common_tokens) /
            smaller_count
        )

        if containment_score == 1.0:
            return 0.96

        token_overlap_score = (
            2 * len(common_tokens)
            / (len(a_tokens) + len(b_tokens))
        )

    else:
        token_overlap_score = 0.0

    # ========================================================
    # DIRECT STRING SIMILARITY
    # ========================================================

    direct_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    # ========================================================
    # TOKEN LEVEL SIMILARITY
    # ========================================================

    token_scores = []

    for token_a in a_tokens:

        best_token_score = 0.0

        for token_b in b_tokens:

            score = SequenceMatcher(
                None,
                token_a,
                token_b
            ).ratio()

            best_token_score = max(
                best_token_score,
                score
            )

        token_scores.append(
            best_token_score
        )

    if token_scores:

        token_similarity_score = (
            sum(token_scores) /
            len(token_scores)
        )

    else:
        token_similarity_score = 0.0

    return max(
        direct_score,
        token_overlap_score,
        token_similarity_score
    )


# ============================================================
# FIND SUPPLIER
# ============================================================

def find_supplier(
    supplier_name: str,
    suppliers
):

    if not supplier_name:
        return None

    requested = str(
        supplier_name
    ).strip()

    if not requested:
        return None

    requested_normalized = normalize_supplier_name(
        requested
    )

    # ========================================================
    # 1. NORMALIZED MATCH FIRST
    #
    # IMPORTANT:
    # Do this BEFORE raw exact match.
    #
    # Example:
    # एबीसी इलेक्ट्रिकल
    # ->
    # abc electrical
    #
    # DB:
    # ABC Electrical
    # एबीसी इलेक्ट्रिकल
    #
    # Prefer the normalized English/Hinglish record.
    # ========================================================

    normalized_matches = []

    if requested_normalized:

        for supplier in suppliers:

            db_name = str(
                supplier.get("name", "")
            ).strip()

            db_normalized = normalize_supplier_name(
                db_name
            )

            if (
                db_normalized
                and
                requested_normalized == db_normalized
            ):
                normalized_matches.append(
                    supplier
                )

    if normalized_matches:

        # Prefer a supplier whose stored name
        # is already English/Hinglish.
        latin_matches = [
            supplier
            for supplier in normalized_matches
            if not contains_devanagari(
                str(supplier.get("name", ""))
            )
        ]

        if latin_matches:
            return latin_matches[0]

        return normalized_matches[0]

    # ========================================================
    # 2. RAW EXACT MATCH
    # ========================================================

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        ).strip()

        if (
            db_name.lower()
            == requested.lower()
        ):
            return supplier

    # ========================================================
    # 3. TOKEN CONTAINMENT
    # ========================================================

    requested_tokens = supplier_tokens(
        requested
    )

    containment_matches = []

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        )

        db_tokens = supplier_tokens(
            db_name
        )

        if not requested_tokens or not db_tokens:
            continue

        common_tokens = (
            requested_tokens
            .intersection(db_tokens)
        )

        if not common_tokens:
            continue

        smaller_count = min(
            len(requested_tokens),
            len(db_tokens)
        )

        containment = (
            len(common_tokens) /
            smaller_count
        )

        if containment == 1.0:

            containment_matches.append(
                supplier
            )

    if len(containment_matches) == 1:
        return containment_matches[0]

    # ========================================================
    # 4. FUZZY MATCHING
    # ========================================================

    matches = []

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        )

        score = supplier_similarity(
            requested,
            db_name
        )

        matches.append(
            (
                score,
                supplier
            )
        )

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_supplier = (
        matches[0]
    )

    # ========================================================
    # AMBIGUITY PROTECTION
    # ========================================================

    if len(matches) > 1:

        second_score = matches[1][0]

        if (
            best_score < 0.90
            and
            (best_score - second_score) < 0.08
        ):
            return None

    # ========================================================
    # HIGH CONFIDENCE FUZZY MATCH
    # ========================================================

    if best_score >= 0.78:
        return best_supplier

    return None


# ============================================================
# DEVANAGARI DETECTION
# ============================================================

def contains_devanagari(value: str) -> bool:

    if not value:
        return False

    return bool(
        re.search(
            r"[\u0900-\u097F]",
            str(value)
        )
    )


# ============================================================
# SEND TRANSACTION
# ============================================================

def send_transaction(
    transaction,
    token: str
):

    print(
        "AI TRANSACTION SUPPLIER:",
        transaction.supplier_name,
        flush=True
    )

    suppliers = get_suppliers(
        token
    )

    print(
        "AI SUPPLIERS RECEIVED:",
        len(suppliers),
        flush=True
    )

    supplier = find_supplier(
        transaction.supplier_name,
        suppliers
    )

    # ========================================================
    # SUPPLIER NOT FOUND
    # ========================================================

    if supplier is None:

        print(
            "SUPPLIER MATCH FAILED:",
            transaction.supplier_name,
            flush=True
        )

        normalized_display_name = (
            normalize_supplier_display_name(
                transaction.supplier_name
            )
        )

        return {
            "status": "SUPPLIER_NOT_FOUND",
            "message": (
                f"Supplier "
                f"'{transaction.supplier_name}' "
                "not found in database"
            ),
            "supplier_name": (
                normalized_display_name
            ),
            "detected_supplier_name": (
                transaction.supplier_name
            )
        }

    # ========================================================
    # SUPPLIER MATCH SUCCESS
    # ========================================================

    print(
        "SUPPLIER MATCH SUCCESS:",
        supplier.get("name"),
        supplier.get("id"),
        flush=True
    )

    # ========================================================
    # IMPORTANT:
    # If the matched supplier is already stored in Hindi,
    # don't silently save a transaction against it when the
    # requested normalized English supplier exists nowhere.
    #
    # Return NOT_FOUND so frontend can create the clean
    # English/Hinglish supplier through confirmation flow.
    # ========================================================

    stored_supplier_name = str(
        supplier.get("name", "")
    ).strip()

    requested_display_name = (
        normalize_supplier_display_name(
            transaction.supplier_name
        )
    )

    if (
        contains_devanagari(stored_supplier_name)
        and
        requested_display_name
        and
        not contains_devanagari(
            requested_display_name
        )
    ):

        print(
            "HINDI SUPPLIER DETECTED:",
            stored_supplier_name,
            "->",
            requested_display_name,
            flush=True
        )

        return {
            "status": "SUPPLIER_NOT_FOUND",
            "message": (
                f"Supplier '{stored_supplier_name}' "
                f"is stored in Hindi. "
                f"Use '{requested_display_name}' "
                f"as the supplier name."
            ),
            "supplier_name": requested_display_name,
            "detected_supplier_name": (
                transaction.supplier_name
            )
        }

    # ========================================================
    # TRANSACTION DATA
    # ========================================================

    data = {
        "supplier_id": supplier["id"],
        "transaction_type": (
            transaction.transaction_type
        ),
        "amount": transaction.amount,
        "transaction_date": (
            transaction.transaction_date.isoformat()
        ),
        "reference_number": (
            transaction.reference_number
        ),
        "notes": transaction.notes
    }

    # ========================================================
    # SEND TO BACKEND
    # ========================================================

    response = requests.post(
        f"{BACKEND_URL}/api/v1/transactions",
        json=data,
        headers=_headers(token),
        timeout=20
    )

    # ========================================================
    # DUPLICATE TRANSACTION
    # ========================================================

    if response.status_code == 409:

        return {
            "status": "DUPLICATE_TRANSACTION",
            "message": (
                "Duplicate transaction detected"
            )
        }

    # ========================================================
    # BACKEND ERRORS
    # ========================================================

    response.raise_for_status()

    return response.json()