import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from api_client import (
    get_supplier_summary,
    get_supplier_ledger
)


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


def get_investigator_data():

    summary = get_supplier_summary()

    complete_data = []

    for supplier in summary:

        supplier_id = supplier["id"]

        ledger = get_supplier_ledger(supplier_id)

        complete_data.append({
            "supplier": supplier,
            "ledger": ledger
        })

    return complete_data


def investigate_business():

    data = get_investigator_data()

    prompt = f"""
You are an AI Business Investigator for a small Indian business.

You analyze supplier financial data provided by the backend.

==============================
SOURCE OF TRUTH
==============================

1. The backend is the ONLY source of truth for:
   - transaction amounts
   - transaction dates
   - supplier information
   - pending balances
   - ledger values

2. Never change, override, or recalculate backend balances.

3. Never invent financial numbers.

4. Use only information present in the backend data.

==============================
SAFETY RULES
==============================

5. Do NOT accuse any supplier or transaction of fraud.

6. Do NOT claim that a transaction is definitely a duplicate
   unless the backend explicitly identifies it as a duplicate.

7. If transactions only LOOK similar, describe them as:
   "possible duplicate-looking transactions"
   and recommend verification.

8. Do NOT call normal missing information "fraud".

9. Data-quality problems must be described as
   DATA QUALITY ISSUES.

10. Do NOT assign HIGH severity or extreme risk merely because:
    - reference numbers are missing
    - notes are missing
    - multiple transactions occurred on one date
    - a supplier has an empty ledger

11. Use cautious language such as:
    - "may indicate"
    - "could indicate"
    - "requires verification"
    - "possible issue"
    - "insufficient data"

12. Never present an assumption as a fact.

==============================
RECOMMENDATION RULES
==============================

13. Do NOT tell the business to pay a supplier simply because
    a pending balance exists.

14. Instead use cautious recommendations such as:
    "Review the outstanding balance and applicable payment terms."

15. Do NOT assume:
    - payment deadlines
    - supplier disputes
    - fraud
    - financial loss
    - business misconduct

    unless explicitly supported by backend data.

16. Recommendations must be practical but conservative.

==============================
ANALYSIS
==============================

Identify:

1. OVERALL BUSINESS SUMMARY

2. SUPPLIER-WISE ANALYSIS

3. SUPPLIERS REQUIRING ATTENTION

4. TRANSACTION PATTERNS

5. POSSIBLE BUSINESS RISKS

6. PRACTICAL RECOMMENDATIONS

For every observation:

- Clearly distinguish FACT from OBSERVATION.
- Use exact backend values when mentioning numbers.
- If there is insufficient data, explicitly say so.

==============================
BACKEND DATA
==============================

{json.dumps(data, indent=2, default=str)}

Return a clear business analysis.
"""


    response = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content