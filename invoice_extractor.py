import os
import base64

from dotenv import load_dotenv
from openai import OpenAI

from schemas import Transaction


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# ==========================================
# PLACEHOLDER SUPPLIER NAME CHECK
# ==========================================

def clean_supplier_name(name):
    if name is None:
        return None

    cleaned = " ".join(name.strip().lower().split())

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


# ==========================================
# INVOICE EXTRACTION
# ==========================================

def extract_invoice(image_path: str) -> Transaction:

    # Read invoice image
    with open(image_path, "rb") as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")


    prompt = """
You are an AI invoice extraction system.

Carefully inspect the ENTIRE invoice image before answering.

Extract information from the invoice and return ONLY valid JSON.

Required JSON fields:

{
    "transaction_type": "PURCHASE",
    "supplier_name": "...",
    "amount": 0,
    "payment_status": null,
    "transaction_date": "YYYY-MM-DD",
    "reference_number": "...",
    "notes": null
}


==========================================
TRANSACTION TYPE
==========================================

1. transaction_type must be one of:

   PURCHASE
   PAYMENT
   RETURN
   CREDIT_NOTE


For a normal purchase invoice, use:

PURCHASE


==========================================
SUPPLIER NAME
==========================================

2. supplier_name:

Extract the REAL company/supplier issuing the invoice.

IMPORTANT:

Some invoice templates contain placeholder text such as:

- "Add Company Name"
- "Add Name"
- "Company Name"
- "Add Company"
- "Your Company Name"
- "Enter Company Name"
- "Supplier Name"

These are NOT real supplier names.

If the invoice only contains placeholder company information,
return:

"supplier_name": null

Do NOT treat placeholder text as the supplier.

Do NOT use:

- customer/buyer name
- billing customer name
- shipping customer name
- transporter name
- bank account name
- placeholder company name


==========================================
AMOUNT
==========================================

3. amount:

THIS IS VERY IMPORTANT.

Find the FINAL TOTAL of the invoice.

Look carefully at the bottom total section.

Use the amount next to:

- "Total"
- "Grand Total"
- "Total Amount"
- "Total Payable"
- equivalent final-total wording

DO NOT use:

- individual item amount
- list price
- discount amount
- tax amount
- subtotal
- amount already paid
- settled amount
- invoice balance

Example:

If the invoice shows:

Total = ₹1,16,800.00

then:

"amount": 116800.00

Remove:

- currency symbols
- commas

from the numeric value.


==========================================
TRANSACTION DATE
==========================================

4. transaction_date:

Extract the INVOICE DATE.

Do NOT use:

- due date
- delivery date
- transporter date
- transporter document date
- E-way bill date
- E-way bill document date

Convert the invoice date to:

YYYY-MM-DD

Example:

22-Apr-25

becomes:

2025-04-22


==========================================
REFERENCE NUMBER
==========================================

5. reference_number:

Extract the INVOICE NUMBER.

Do NOT use:

- IRN
- Ack No.
- transporter document number
- E-way bill number

Example:

Invoice Number: PPP/0001/25-26

becomes:

"reference_number": "PPP/0001/25-26"


==========================================
PAYMENT STATUS
==========================================

6. payment_status:

Extract it only if clearly mentioned.

If it is not clearly mentioned:

"payment_status": null


==========================================
NOTES
==========================================

7. notes:

Include useful additional invoice information only if necessary.

Otherwise:

"notes": null


==========================================
GENERAL SAFETY
==========================================

8. Do NOT invent information.

If a field truly does not exist, return null.

9. Understand:

- Indian invoices
- GST invoices
- Indian Rupee (₹)
- Indian date formats
- English
- Hindi
- Hinglish

10. Do NOT calculate pending balances.

11. Do NOT modify any database.


==========================================
FINAL INSPECTION
==========================================

12. Before producing the JSON, carefully inspect:

- invoice header
- supplier/company name
- invoice number
- invoice date
- complete item section
- discount
- taxes
- final total section
- payment/settlement information

13. The FINAL TOTAL is more important than any
individual amount shown elsewhere on the invoice.

14. If the company name is only placeholder text,
supplier_name MUST be null.

15. Return ONLY the JSON object.
"""


    response = client.chat.completions.create(

        model="google/gemini-2.5-flash",

        max_tokens=200,

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


    # Get AI response
    data = response.choices[0].message.content


    # Convert JSON → Transaction
    transaction = Transaction.model_validate_json(data)


    # ==========================================
    # FINAL SUPPLIER SAFETY CHECK
    # ==========================================

    transaction.supplier_name = clean_supplier_name(
        transaction.supplier_name
    )


    return transaction