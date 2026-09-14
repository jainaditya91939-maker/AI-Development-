from datetime import date

from ai_extractor import extract_transaction
from invoice_extractor import extract_invoice
from api_client import send_transaction


def process_transaction(text: str):
    transaction = extract_transaction(text)

    # If the user did not mention a date,
    # use today's date for a voice transaction.
    if transaction.transaction_date is None:
        transaction.transaction_date = date.today()

    missing_fields = []

    if transaction.supplier_name is None:
        missing_fields.append("supplier name")

    if transaction.amount is None:
        missing_fields.append("amount")

    if missing_fields:
        return {
            "status": "NEEDS_INFORMATION",
            "message": "Missing required information",
            "missing_fields": missing_fields,
            "transaction": transaction.model_dump(mode="json")
        }

    result = send_transaction(transaction)

    # Supplier does not exist
    if result.get("status") == "SUPPLIER_NOT_FOUND":
        return result

    # Duplicate transaction
    if result.get("status") == "DUPLICATE_TRANSACTION":
        return result

    # Transaction successfully saved
    return {
        "status": "SUCCESS",
        "message": "Transaction saved successfully",
        "transaction": result["transaction"]
    }


def process_invoice(image_path: str):
    transaction = extract_invoice(image_path)

    missing_fields = []

    if transaction.supplier_name is None:
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
            "transaction": transaction.model_dump(mode="json")
        }

    result = send_transaction(transaction)

    # Supplier does not exist
    if result.get("status") == "SUPPLIER_NOT_FOUND":
        return result

    # Duplicate transaction
    if result.get("status") == "DUPLICATE_TRANSACTION":
        return result

    # Invoice transaction successfully saved
    return {
        "status": "SUCCESS",
        "message": "Invoice transaction saved successfully",
        "transaction": result["transaction"]
    }