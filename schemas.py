from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date


class Transaction(BaseModel):
    transaction_type: Literal[
        "PURCHASE",
        "PAYMENT",
        "RETURN",
        "CREDIT_NOTE"
    ]

    supplier_name: Optional[str] = None

    amount: Optional[float] = Field(
        default=None,
        ge=0
    )

    payment_status: Optional[str] = None

    transaction_date: Optional[date] = None

    reference_number: Optional[str] = None

    notes: Optional[str] = None