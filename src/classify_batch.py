import sys
import os
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import classify_email

INPUT_CSV = "data/raw_emails.csv"
OUTPUT_CSV = "data/predictions.csv"


def main():
    print("📄 Loading input CSV...")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"subject", "body"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain 'subject' and 'body' columns")

    predictions = []

    print("🤖 Classifying emails using local Mistral LLM...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = classify_email(
                subject=str(row["subject"]),
                body=str(row["body"])
            )
        except Exception as e:
            print(f"⚠️ Classification failed for one row: {e}")
            result = {
                "intent": "Other",
                "priority": "Low",
                "sentiment": "Neutral"
            }

        predictions.append(result)

    result_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(predictions)],
        axis=1
    )

    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Batch classification completed")
    print(f"📁 Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
