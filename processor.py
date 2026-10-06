from datetime import date

from ai_extractor import extract_transaction
from invoice_extractor import extract_invoice
from api_client import send_transaction


def process_transaction(text: str, token: str):

    try:
        transaction = extract_transaction(text)

    except ValueError as e:
        return {
            "status": "NEEDS_INFORMATION",
            "message": str(e),
            "missing_fields": [],
        }

    except Exception as e:
        print(
            "PROCESS TRANSACTION EXTRACTION ERROR:",
            repr(e),
            flush=True,
        )

        return {
            "status": "ERROR",
            "message": (
                "I could not understand this voice command. "
                "Please say supplier name, amount and transaction type clearly."
            ),
        }

    # Voice transaction uses today's date when
    # the user did not explicitly mention a date.
    if transaction.transaction_date is None:
        transaction.transaction_date = date.today()

    missing_fields = []

    if not transaction.supplier_name:
        missing_fields.append("supplier name")

    if transaction.amount is None:
        missing_fields.append("amount")

    if not transaction.transaction_type:
        missing_fields.append("transaction type")

    if missing_fields:
        return {
            "status": "NEEDS_INFORMATION",
            "message": (
                "Please provide the missing transaction information."
            ),
            "missing_fields": missing_fields,
            "transaction": transaction.model_dump(mode="json"),
        }

    try:
        result = send_transaction(
            transaction,
            token,
        )

    except ValueError as e:
        print(
            "TRANSACTION VALUE ERROR:",
            repr(e),
            flush=True,
        )

        return {
            "status": "ERROR",
            "message": str(e),
        }

    except Exception as e:
        print(
            "TRANSACTION SEND ERROR:",
            repr(e),
            flush=True,
        )

        return {
            "status": "ERROR",
            "message": (
                "I could not save this transaction right now. "
                "Please try again."
            ),
        }

    if result.get("status") == "SUPPLIER_NOT_FOUND":
        return result

    if result.get("status") == "DUPLICATE_TRANSACTION":
        return result

    if result.get("status") == "ERROR":
        return result

    if "transaction" not in result:
        return {
            "status": "ERROR",
            "message": "Transaction was not confirmed by the backend.",
        }

    return {
        "status": "SUCCESS",
        "message": "Transaction saved successfully",
        "transaction": result["transaction"],
    }


def process_invoice(image_path: str, token: str):

    try:
        transaction = extract_invoice(image_path)

    except Exception as e:
        print(
            "INVOICE EXTRACTION ERROR:",
            repr(e),
            flush=True,
        )

        return {
            "status": "ERROR",
            "message": (
                "I could not read this invoice. "
                "Please try a clearer invoice image."
            ),
        }

    missing_fields = []

    if not transaction.supplier_name:
        missing_fields.append("supplier name")

    if transaction.amount is None:
        missing_fields.append("amount")

    if transaction.transaction_date is None:
        missing_fields.append("transaction date")

    if missing_fields:
        return {
            "status": "NEEDS_INFORMATION",
            "message": "Missing required information from invoice",
            "missing_fields": missing_fields,
            "transaction": transaction.model_dump(mode="json"),
        }

    try:
        result = send_transaction(
            transaction,
            token,
        )

    except Exception as e:
        print(
            "INVOICE TRANSACTION SEND ERROR:",
            repr(e),
            flush=True,
        )

        return {
            "status": "ERROR",
            "message": (
                "Invoice was read, but the transaction could not be saved."
            ),
        }

    if result.get("status") == "SUPPLIER_NOT_FOUND":
        return result

    if result.get("status") == "DUPLICATE_TRANSACTION":
        return result

    if result.get("status") == "ERROR":
        return result

    if "transaction" not in result:
        return {
            "status": "ERROR",
            "message": "Invoice transaction was not confirmed by backend.",
        }

    return {
        "status": "SUCCESS",
        "message": "Invoice transaction saved successfully",
        "transaction": result["transaction"],
    }