import os
import re
import requests
from difflib import SequenceMatcher


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


# ============================================================
# AUTHENTICATION
# ============================================================

def _headers(token: str):
    if not token:
        raise ValueError("Authentication token is required")

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


# ============================================================
# SUPPLIERS API
# ============================================================

def get_suppliers(token: str):

    response = requests.get(
        f"{BACKEND_URL}/api/v1/suppliers",
        headers=_headers(token),
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        return []

    return data


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
# ENGLISH + ROMAN HINGLISH ONLY
# ============================================================

def normalize_supplier_name(name: str) -> str:

    if not name:
        return ""

    value = str(name).lower().strip()

    # Common speech-recognition variations
    replacements = {
        "havel": "havells",
        "havell": "havells",
        "havels": "havells",
        "havells": "havells",

        "poly cab": "polycab",
        "polycab": "polycab",

        "finolex": "finolex",

        "anchor": "anchor",

        "philips": "philips",

        "orient": "orient",

        "bajaj": "bajaj",

        "usha": "usha",

        "crompton": "crompton",

        "wipro": "wipro"
    }

    value = replacements.get(value, value)

    # Remove anything except English letters,
    # numbers, spaces, dot, &, and hyphen.
    value = re.sub(
        r"[^a-z0-9\s.&-]",
        " ",
        value
    )

    # Remove common company suffixes
    value = re.sub(
        r"\b(private limited|pvt ltd|pvt\. ltd\.|private ltd|limited|ltd|llp|incorporated|inc)\b",
        " ",
        value
    )

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

    if a == b:
        return 1.0

    a_tokens = supplier_tokens(name1)
    b_tokens = supplier_tokens(name2)

    if not a_tokens or not b_tokens:
        return 0.0

    common_tokens = a_tokens.intersection(
        b_tokens
    )

    if common_tokens:

        smaller_count = min(
            len(a_tokens),
            len(b_tokens)
        )

        containment_score = (
            len(common_tokens)
            / smaller_count
        )

        if containment_score == 1.0:
            return 0.96

        token_overlap_score = (
            2 * len(common_tokens)
            / (len(a_tokens) + len(b_tokens))
        )

    else:
        token_overlap_score = 0.0

    direct_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

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
            sum(token_scores)
            / len(token_scores)
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

        if db_name.lower() == requested.lower():
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
            requested_normalized == db_normalized
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
            requested_tokens.intersection(
                db_tokens
            )
        )

        if not common_tokens:
            continue

        smaller_count = min(
            len(requested_tokens),
            len(db_tokens)
        )

        containment = (
            len(common_tokens)
            / smaller_count
        )

        if containment == 1.0:

            containment_matches.append(
                supplier
            )

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

    # Ambiguity protection
    if len(matches) > 1:

        second_score = matches[1][0]

        if (
            best_score < 0.90
            and
            (best_score - second_score) < 0.08
        ):
            return None

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

    # --------------------------------------------------------
    # Get suppliers using the SAME JWT.
    # Backend therefore returns only the current
    # user's business suppliers.
    # --------------------------------------------------------

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
    # Validate amount
    # --------------------------------------------------------

    if transaction.amount is None:

        return {
            "status": "NEEDS_INFORMATION",
            "message": "Transaction amount is required"
        }

    if float(transaction.amount) <= 0:

        return {
            "status": "NEEDS_INFORMATION",
            "message": "Transaction amount must be greater than zero"
        }

    # --------------------------------------------------------
    # Transaction date
    # --------------------------------------------------------

    if transaction.transaction_date:

        transaction_date = (
            transaction.transaction_date.isoformat()
        )

    else:

        from datetime import date

        transaction_date = date.today().isoformat()

    # --------------------------------------------------------
    # Transaction data
    # --------------------------------------------------------

    data = {
        "supplier_id": supplier["id"],
        "transaction_type": (
            transaction.transaction_type
        ),
        "amount": float(transaction.amount),
        "transaction_date": transaction_date,
        "reference_number": (
            transaction.reference_number
        ),
        "notes": transaction.notes
    }

    print(
        "AI TRANSACTION PAYLOAD:",
        data,
        flush=True
    )

    # --------------------------------------------------------
    # Send to backend
    # --------------------------------------------------------

    response = requests.post(
        f"{BACKEND_URL}/api/v1/transactions",
        json=data,
        headers=_headers(token),
        timeout=20
    )

    print(
        "BACKEND TRANSACTION STATUS:",
        response.status_code,
        flush=True
    )

    # --------------------------------------------------------
    # Duplicate transaction
    # --------------------------------------------------------

    if response.status_code == 409:

        return {
            "status": "DUPLICATE_TRANSACTION",
            "message": "Duplicate transaction detected"
        }

    # --------------------------------------------------------
    # Authentication failure
    # --------------------------------------------------------

    if response.status_code == 401:

        return {
            "status": "AUTHENTICATION_ERROR",
            "message": "Authentication token is invalid or expired"
        }

    # --------------------------------------------------------
    # Other backend errors
    # --------------------------------------------------------

    response.raise_for_status()

    result = response.json()

    print(
        "BACKEND TRANSACTION RESULT:",
        result,
        flush=True
    )

    return result