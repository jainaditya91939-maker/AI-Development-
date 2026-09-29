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
# SUPPLIER NAME NORMALIZATION
# ============================================================

def normalize_supplier_name(name: str) -> str:

    if not name:
        return ""

    value = str(name).lower().strip()

    # --------------------------------------------------------
    # Common Hindi speech-recognition aliases
    # --------------------------------------------------------

    hindi_aliases = {

        # ====================================================
        # HAVELLS
        # ====================================================

        "हैवेल्स": "havells",
        "हैवेल": "havells",
        "हैवल्स": "havells",
        "हैवल": "havells",
        "हैवेलस": "havells",
        "हैवन्स": "havells",
        "हैवंत": "havells",

        # ====================================================
        # POLYCAB
        # ====================================================

        "पॉलीकैब": "polycab",
        "पॉली कैब": "polycab",
        "पोलिकैब": "polycab",
        "पोलीकैब": "polycab",
        "पोली कैब": "polycab",

        # New speech-recognition variants
        "होलीकैब": "polycab",
        "होली कैब": "polycab",
        "होलीकेब": "polycab",
        "होली केब": "polycab",
        "पॉली केब": "polycab",
        "पोलि कैब": "polycab",

        # ====================================================
        # ANCHOR
        # ====================================================

        "एंकर": "anchor",

        # ====================================================
        # FINOLEX
        # ====================================================

        "फिनोलेक्स": "finolex",

        # ====================================================
        # RR KABEL
        # ====================================================

        "आरआर": "rr",
        "आर आर": "rr",
        "आरआरकेबल": "rr kabel",
        "आरआर केबल": "rr kabel",
        "आर आर केबल": "rr kabel",
        "केबल": "cable",

        # ====================================================
        # SCHNEIDER
        # ====================================================

        "श्नाइडर": "schneider",
        "स्नाइडर": "schneider",

        # ====================================================
        # PHILIPS
        # ====================================================

        "फिलिप्स": "philips",
        "फिलिप": "philips",

        # ====================================================
        # CROMPTON
        # ====================================================

        "क्रॉम्पटन": "crompton",
        "क्रॉम्प्टन": "crompton",

        # ====================================================
        # LEGRAND
        # ====================================================

        "लेग्रैंड": "legrand",
        "लेग्रां": "legrand",

        # ====================================================
        # WIPRO
        # ====================================================

        "विप्रो": "wipro",

        # ====================================================
        # BAJAJ
        # ====================================================

        "बजाज": "bajaj",

        # ====================================================
        # COMMON ELECTRICAL WORDS
        # ====================================================

        "इलेक्ट्रिकल": "electrical",
        "इलेक्ट्रिकल्स": "electricals",
        "इलेक्ट्रिक": "electric",
        "इलेक्ट्रिक्स": "electrics",
        "एलईडी": "led",
        "डीसी": "dc",
        "एसी": "ac",
    }

    # Apply Hindi aliases
    for hindi_word, english_word in hindi_aliases.items():
        value = value.replace(
            hindi_word,
            english_word
        )

    # --------------------------------------------------------
    # Common company suffixes
    # --------------------------------------------------------

    value = re.sub(
        r"\b(private limited|pvt ltd|pvt\. ltd\.|"
        r"private ltd|limited|ltd|llp|incorporated|inc)\b",
        " ",
        value
    )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


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

    # --------------------------------------------------------
    # Exact normalized match
    # --------------------------------------------------------

    if a == b:
        return 1.0

    a_tokens = supplier_tokens(name1)
    b_tokens = supplier_tokens(name2)

    if not a_tokens or not b_tokens:
        return 0.0

    # --------------------------------------------------------
    # Exact token overlap
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Direct string similarity
    # --------------------------------------------------------

    direct_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    # --------------------------------------------------------
    # Token-level similarity
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------------

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        ).strip()

        if (
            db_name.lower()
            == requested.lower()
        ):
            return supplier

    # --------------------------------------------------------
    # 2. Normalized exact match
    # --------------------------------------------------------

    requested_normalized = (
        normalize_supplier_name(
            requested
        )
    )

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        )

        db_normalized = (
            normalize_supplier_name(
                db_name
            )
        )

        if (
            requested_normalized
            and
            requested_normalized
            == db_normalized
        ):
            return supplier

    # --------------------------------------------------------
    # 3. Token containment
    # --------------------------------------------------------

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

    # Only use containment when there is
    # exactly one clear supplier.
    if len(containment_matches) == 1:
        return containment_matches[0]

    # --------------------------------------------------------
    # 4. Fuzzy matching
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ambiguity protection
    # --------------------------------------------------------

    if len(matches) > 1:

        second_score = matches[1][0]

        if (
            best_score < 0.90
            and
            (best_score - second_score) < 0.08
        ):
            return None

    # --------------------------------------------------------
    # High-confidence fuzzy match
    # --------------------------------------------------------

    if best_score >= 0.78:
        return best_supplier

    return None


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

    # --------------------------------------------------------
    # Supplier not found
    # --------------------------------------------------------

    if supplier is None:

        print(
            "SUPPLIER MATCH FAILED:",
            transaction.supplier_name,
            flush=True
        )

        return {
            "status": "SUPPLIER_NOT_FOUND",
            "message": (
                f"Supplier "
                f"'{transaction.supplier_name}' "
                "not found in database"
            ),
            "supplier_name": (
                transaction.supplier_name
            )
        }

    print(
        "SUPPLIER MATCH SUCCESS:",
        supplier.get("name"),
        supplier.get("id"),
        flush=True
    )

    # --------------------------------------------------------
    # Transaction data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Send to backend
    # --------------------------------------------------------

    response = requests.post(
        f"{BACKEND_URL}/api/v1/transactions",
        json=data,
        headers=_headers(token),
        timeout=20
    )

    # --------------------------------------------------------
    # Duplicate transaction
    # --------------------------------------------------------

    if response.status_code == 409:

        return {
            "status": "DUPLICATE_TRANSACTION",
            "message": (
                "Duplicate transaction detected"
            )
        }

    # --------------------------------------------------------
    # Backend errors
    # --------------------------------------------------------

    response.raise_for_status()

    return response.json()