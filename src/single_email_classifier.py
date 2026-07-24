import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample email
subject = "Payment failed"
body = "My payment failed but the amount was deducted from my bank account."

# Prompt
prompt = f"""
You are a customer support AI assistant.

Analyze the email below and classify it into:
- Intent (Complaint, Request, Query, Feedback, Other)
- Priority (High, Medium, Low)
- Sentiment (Positive, Neutral, Negative)

Respond ONLY in JSON format like this:
{{
  "intent": "",
  "priority": "",
  "sentiment": ""
}}

Email Subject: {subject}
Email Body: {body}
"""

# Call the LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# Print result
print("LLM response:")
print(response.choices[0].message.content)
