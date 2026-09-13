
from schemas import Transaction


def mock_extract_transaction(text: str) -> Transaction:

    text_lower = text.lower()

    # PAYMENT
    if "payment" in text_lower or "diya" in text_lower or "diye" in text_lower:

        return Transaction(
            transaction_type="payment",
            supplier_name="Sharma Electrical",
            amount=20000
        )

    # RETURN
    elif "return" in text_lower or "wapas" in text_lower:

        return Transaction(
            transaction_type="return",
            supplier_name="Gupta Electrical",
            amount=15000
        )

    # PURCHASE
    elif "maal" in text_lower or "material" in text_lower or "kharida" in text_lower:

        return Transaction(
            transaction_type="purchase",
            supplier_name="Sharma Electrical",
            amount=50000,
            payment_status="credit"
        )

    # UNKNOWN
    else:

        return Transaction(
            transaction_type="purchase",
            supplier_name=None,
            amount=None
        )

