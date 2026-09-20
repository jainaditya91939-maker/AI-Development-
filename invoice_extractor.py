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


MODEL = "google/gemma-4-26b-a4b-it:free"


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
You are an Indian GST invoice extraction system.

Carefully inspect the complete invoice image.

Return ONLY one JSON object.

Use exactly these fields:

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

1. supplier_name:
   Extract the SELLER / ISSUER of the invoice.
   Do NOT use the buyer/customer name.

2. amount:
   Extract the FINAL GRAND TOTAL / TOTAL PAYABLE.
   Do NOT use subtotal.
   Do NOT use an individual item amount.
   Do NOT use only the tax amount.

3. transaction_date:
   Extract the INVOICE DATE.
   Do NOT use due date or delivery date.
   Convert it to YYYY-MM-DD.

4. reference_number:
   Extract the INVOICE NUMBER.
   Do NOT use IRN, Ack No. or e-way bill number.

5. transaction_type:
   Use PURCHASE for a normal invoice.

6. payment_status:
   Only extract it when explicitly written.
   Otherwise use null.

7. notes:
   Use null unless useful information is clearly present.

8. Never invent information.
   Use null when something cannot be read.

9. Do not calculate balances.

10. Do not modify any database.

Important:
Seller = supplier.
Invoice date = transaction date.
Grand total = transaction amount.
Invoice number = reference number.
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
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
            "INVOICE_V2: AI returned no choices"
        )

    choice = response.choices[0]

    data = choice.message.content

    print(
        "INVOICE_V2 MODEL:",
        MODEL
    )

    print(
        "INVOICE_V2 FINISH:",
        choice.finish_reason
    )

    print(
        "INVOICE_V2 CONTENT:",
        repr(data)
    )

    if not data:

        print(
            "INVOICE_V2 RAW RESPONSE:",
            response.model_dump()
        )

        raise ValueError(
            "INVOICE_V2: AI returned empty content"
        )

    json_text = extract_json(
        data
    )

    if json_text is None:

        print(
            "INVOICE_V2 INVALID CONTENT:",
            repr(data)
        )

        raise ValueError(
            "INVOICE_V2: AI response was not valid JSON"
        )

    transaction = Transaction.model_validate_json(
        json_text
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