import os
import base64
import json
import mimetypes
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


MODEL = "inclusionai/ling-3.0-flash-vl:free"


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

    return bool(
        re.match(
            r"^\d{4}-\d{2}-\d{2}$",
            str(value)
        )
    )


def get_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        return mime_type

    return "image/png"


def extract_json(text):
    if not text:
        return None

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:
        candidate = match.group(0)

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def extract_invoice(image_path: str) -> Transaction:

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    mime_type = get_mime_type(
        image_path
    )

    prompt = """
Read this invoice image carefully.

Return ONLY valid JSON.
Do not write anything before or after the JSON.

Required format:

{
  "transaction_type": "PURCHASE",
  "supplier_name": null,
  "amount": null,
  "payment_status": null,
  "transaction_date": null,
  "reference_number": null,
  "notes": null
}

Extraction rules:

supplier_name:
Use the seller/company that issued the invoice.
Do NOT use the buyer/customer.

amount:
Use the final GRAND TOTAL or TOTAL PAYABLE.
Do NOT use subtotal or tax-only amount.

transaction_date:
Use the invoice date.
Convert it to YYYY-MM-DD.

reference_number:
Use the invoice number.
Do NOT use IRN, Ack number or e-way bill number.

transaction_type:
Use PURCHASE for a normal sales invoice.

payment_status:
Use only if explicitly written.
Otherwise null.

notes:
Use null unless useful information is clearly visible.

Never invent information.
If something cannot be read, use null.

Example:

{
  "transaction_type": "PURCHASE",
  "supplier_name": "Sharma Electricals",
  "amount": 35931,
  "payment_status": null,
  "transaction_date": "2025-09-12",
  "reference_number": "SE/25-26/0147",
  "notes": null
}
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
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
                                f"data:{mime_type};base64,"
                                f"{image_data}"
                            )
                        }
                    }
                ]
            }
        ]
    )

    if not response.choices:
        raise ValueError(
            "Invoice AI returned no choices"
        )

    choice = response.choices[0]

    data = choice.message.content

    print(
        "INVOICE MODEL:",
        MODEL
    )

    print(
        "INVOICE FINISH:",
        choice.finish_reason
    )

    print(
        "INVOICE CONTENT:",
        repr(data)
    )

    if not data:
        raise ValueError(
            "Invoice AI returned empty content"
        )

    json_text = extract_json(
        data
    )

    if json_text is None:
        raise ValueError(
            "Invoice AI returned invalid JSON"
        )

    transaction = (
        Transaction.model_validate_json(
            json_text
        )
    )

    transaction.supplier_name = (
        clean_supplier_name(
            transaction.supplier_name
        )
    )

    if transaction.transaction_date:

        if not has_valid_date(
            transaction.transaction_date
        ):
            transaction.transaction_date = None

    if transaction.amount is not None:

        if transaction.amount <= 0:
            transaction.amount = None

    return transaction