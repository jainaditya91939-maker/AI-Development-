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


MODEL = "google/gemma-4-31b-it:free"


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


def get_image_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if mime_type in allowed_types:
        return mime_type

    return "image/jpeg"


def extract_json_text(text):
    if not text:
        return None

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
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
            return None

    return None


def extract_invoice(image_path: str) -> Transaction:

    mime_type = get_image_mime_type(
        image_path
    )

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    prompt = """
Extract information from this Indian GST invoice.

Return ONLY JSON.

Required fields:

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

- supplier_name = seller/company that issued the invoice.
- Do NOT use the buyer/customer as supplier.
- amount = final grand total / total payable.
- Do NOT use subtotal or individual item amount.
- transaction_date = invoice date only.
- Convert date to YYYY-MM-DD.
- reference_number = invoice number.
- Do NOT use IRN, Ack No. or e-way bill number.
- payment_status = only if explicitly stated.
- Use null when information is unclear.
- Never invent information.
- transaction_type should normally be PURCHASE.
- Do not calculate balances.
- Do not modify any database.

Inspect the complete invoice image carefully.
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
        ],
        response_format={
            "type": "json_object"
        }
    )

    choice = response.choices[0]

    message = choice.message

    data = message.content

    print(
        "Invoice model:",
        MODEL
    )

    print(
        "Invoice finish reason:",
        getattr(
            choice,
            "finish_reason",
            None
        )
    )

    print(
        "Invoice response content:",
        repr(data)
    )

    if not data:

        print(
            "Invoice full response:",
            response.model_dump()
        )

        raise ValueError(
            "Invoice AI returned an empty response. "
            "Check Render logs for the full model response."
        )

    json_text = extract_json_text(
        data
    )

    if json_text is None:

        print(
            "Invoice non-JSON response:",
            repr(data)
        )

        raise ValueError(
            "Invoice AI returned invalid JSON."
        )

    transaction = Transaction.model_validate_json(
        json_text
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