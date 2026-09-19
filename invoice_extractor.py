import os
import base64
import re

from dotenv import load_dotenv
from openai import OpenAI
from schemas import Transaction


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def clean_supplier_name(name):
    if name is None:
        return None

    cleaned = " ".join(
        name.strip().lower().split()
    )

    placeholder_names = {
        "add company name",
        "add name",
        "company name",
        "add company",
        "supplier name",
        "your company name",
        "your company",
        "enter company name",
        "enter company",
        "enter supplier name",
        "company",
        "supplier"
    }

    if cleaned in placeholder_names:
        return None

    return name.strip()


def has_valid_date(value):
    if not value:
        return False

    pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}$"
    )

    return bool(
        pattern.match(str(value))
    )


def extract_invoice(image_path: str) -> Transaction:

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    prompt = """
You are an invoice extraction system.

Carefully inspect the ENTIRE invoice image.

Return ONLY valid JSON with exactly these fields:

{
  "transaction_type": "PURCHASE",
  "supplier_name": null,
  "amount": null,
  "payment_status": null,
  "transaction_date": null,
  "reference_number": null,
  "notes": null
}

Rules:

1. transaction_type:
   Use PURCHASE for a normal invoice.
   Other allowed values:
   PAYMENT, RETURN, CREDIT_NOTE.

2. supplier_name:
   Extract the REAL SELLER / SUPPLIER
   that issued the invoice.

   Do NOT use the buyer/customer name.

   Do NOT use placeholder text such as:
   "Add Company Name",
   "Company Name",
   "Supplier Name",
   "Your Company",
   "Enter Company Name".

   If unclear, use null.

3. amount:
   Extract the FINAL GRAND TOTAL / TOTAL PAYABLE.

   Do NOT use:
   subtotal,
   individual item amount,
   tax amount,
   discount.

   Return only the numeric value.

4. transaction_date:
   Extract the INVOICE DATE.

   Do NOT use:
   due date,
   delivery date,
   e-way bill date.

   Convert the date to:

   YYYY-MM-DD

   If unclear, use null.

5. reference_number:
   Extract the INVOICE NUMBER.

   Do NOT use:
   IRN,
   Ack No.,
   e-way bill number.

   If unclear, use null.

6. payment_status:
   Extract only if clearly written.
   Otherwise use null.

7. notes:
   Add useful invoice information only.
   Otherwise use null.

8. Never invent information.

9. Do not calculate balances.

10. Do not modify any database.

11. Understand Indian GST invoices,
    INR currency and Indian company names.

IMPORTANT:

The seller/issuer is the supplier.

The invoice date is the transaction date.

The final grand total is the transaction amount.

The invoice number is the reference number.
"""

    response = client.chat.completions.create(
        model="google/gemma-4-26b-a4b-it:free",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{image_data}"
                            )
                        }
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    data = response.choices[0].message.content

    if not data:
        raise ValueError(
            "Invoice extractor returned empty response"
        )

    transaction = Transaction.model_validate_json(
        data
    )

    transaction.supplier_name = clean_supplier_name(
        transaction.supplier_name
    )

    if transaction.transaction_date is not None:

        if not has_valid_date(
            transaction.transaction_date
        ):
            transaction.transaction_date = None

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    return transaction