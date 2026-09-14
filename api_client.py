import os
import re
import requests
from difflib import SequenceMatcher


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


# ==========================================
# BASIC API FUNCTIONS
# ==========================================

def get_suppliers():
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers"
    )
    response.raise_for_status()
    return response.json()


def get_supplier_summary():
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers/summary"
    )
    response.raise_for_status()
    return response.json()


def get_supplier_ledger(supplier_id):
    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers/{supplier_id}/ledger"
    )
    response.raise_for_status()
    return response.json()


# ==========================================
# SUPPLIER NAME NORMALIZATION
# ==========================================

def normalize_supplier_name(name: str) -> str:
    """
    Normalize supplier names so that small
    English/Hindi/Hinglish variations can match.
    """

    if not name:
        return ""

    value = name.lower().strip()

    # ------------------------------------------
    # Common Hindi speech-recognition patterns
    # ------------------------------------------

    hindi_aliases = {
        "एबीसी": "abc",
        "ए बी सी": "abc",
        "इलेक्ट्रिकल": "electrical",
        "इलेक्ट्रिकल्स": "electricals",
        "इलेक्ट्रिक": "electric",
        "इलेक्ट्रिक्स": "electrics",
        "एलईडी": "led",
        "डीसी": "dc",
        "एसी": "ac",
    }

    for hindi_word, english_word in hindi_aliases.items():
        value = value.replace(
            hindi_word,
            english_word
        )

    # ------------------------------------------
    # Remove punctuation
    # ------------------------------------------

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    # ------------------------------------------
    # Normalize whitespace
    # ------------------------------------------

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ==========================================
# REMOVE COMMON PLURAL DIFFERENCES
# ==========================================

def supplier_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two supplier names.
    Handles small differences such as:
    Electrical
    Electricals
    """

    a = normalize_supplier_name(name1)
    b = normalize_supplier_name(name2)

    if not a or not b:
        return 0.0

    # Exact normalized match

    if a == b:
        return 1.0

    # Direct similarity

    direct_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    # ------------------------------------------
    # Token comparison
    # ------------------------------------------

    a_tokens = a.split()
    b_tokens = b.split()

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
        token_score = sum(token_scores) / len(token_scores)
    else:
        token_score = 0.0

    return max(
        direct_score,
        token_score
    )


# ==========================================
# FIND SUPPLIER
# ==========================================

def find_supplier(supplier_name: str, suppliers):
    """
    Find the best matching supplier.

    Priority:
    1. Exact name
    2. Normalized exact name
    3. High-confidence fuzzy match
    """

    if not supplier_name:
        return None

    requested = supplier_name.strip()

    # ------------------------------------------
    # 1. Exact match
    # ------------------------------------------

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        ).strip()

        if db_name.lower() == requested.lower():
            return supplier

    # ------------------------------------------
    # 2. Normalized exact match
    # ------------------------------------------

    requested_normalized = normalize_supplier_name(
        requested
    )

    for supplier in suppliers:

        db_name = str(
            supplier.get("name", "")
        )

        db_normalized = normalize_supplier_name(
            db_name
        )

        if (
            requested_normalized
            and requested_normalized == db_normalized
        ):
            return supplier

    # ------------------------------------------
    # 3. Fuzzy matching
    # ------------------------------------------

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

    best_score, best_supplier = matches[0]

    # ------------------------------------------
    # Safety threshold
    # ------------------------------------------

    if best_score >= 0.78:
        return best_supplier

    return None


# ==========================================
# SEND TRANSACTION
# ==========================================

def send_transaction(transaction):

    suppliers = get_suppliers()

    supplier = find_supplier(
        transaction.supplier_name,
        suppliers
    )

    # ------------------------------------------
    # Supplier not found
    # ------------------------------------------

    if supplier is None:

        return {
            "status": "SUPPLIER_NOT_FOUND",
            "message": (
                f"Supplier '{transaction.supplier_name}' "
                "not found in database"
            ),
            "supplier_name": transaction.supplier_name
        }

    # ------------------------------------------
    # Transaction data
    # ------------------------------------------

    data = {
        "supplier_id": supplier["id"],
        "transaction_type": transaction.transaction_type,
        "amount": transaction.amount,
        "transaction_date": (
            transaction.transaction_date.isoformat()
        ),
        "reference_number": transaction.reference_number,
        "notes": transaction.notes
    }

    # ------------------------------------------
    # Send to backend
    # ------------------------------------------

    response = requests.post(
        f"{BACKEND_URL}/api/v1/transactions",
        json=data
    )

    # ------------------------------------------
    # Duplicate transaction
    # ------------------------------------------

    if response.status_code == 409:

        return {
            "status": "DUPLICATE_TRANSACTION",
            "message": "Duplicate transaction detected"
        }

    # ------------------------------------------
    # Other backend errors
    # ------------------------------------------

    response.raise_for_status()

    return response.json()