import sys
import os
import streamlit as st
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import classify_email


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Email Intent & Priority Classifier",
    layout="centered"
)

st.title("📧 Email Intent & Priority Classifier")
st.write(
    "This application uses a **local Mistral LLM (via Ollama)** to classify emails "
    "into **intent, priority, and sentiment**. "
    "No cloud APIs or rule-based fallbacks are used."
)

st.divider()

# --------------------------------------------------
# SINGLE EMAIL CLASSIFICATION
# --------------------------------------------------
st.header("✉️ Single Email Classification")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body", height=160)

if st.button("Classify Single Email"):
    if not subject.strip() or not body.strip():
        st.warning("Please enter both subject and body.")
    else:
        try:
            result = classify_email(subject, body)

            st.subheader("🧠 Classification Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Intent", result["intent"])
            with col2:
                st.metric("Priority", result["priority"])
            with col3:
                st.metric("Sentiment", result["sentiment"])

        except Exception as e:
            st.error("❌ Classification failed")
            st.caption(
                "The local Mistral model did not return a valid structured response."
            )
            st.code(str(e))

st.divider()

# --------------------------------------------------
# CSV BATCH CLASSIFICATION
# --------------------------------------------------
st.header("📂 CSV Batch Classification")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain `subject` and `body` columns)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if not {"subject", "body"}.issubset(df.columns):
            st.error("CSV must contain 'subject' and 'body' columns.")
        else:
            st.success(f"Loaded {len(df)} emails")

            if st.button("Classify CSV"):
                results = []

                with st.spinner("Classifying emails using local Mistral LLM..."):
                    for _, row in df.iterrows():
                        try:
                            res = classify_email(
                                str(row["subject"]),
                                str(row["body"])
                            )
                        except Exception:
                            res = {
                                "intent": "Other",
                                "priority": "Low",
                                "sentiment": "Neutral"
                            }
                        results.append(res)

                result_df = pd.concat(
                    [df.reset_index(drop=True), pd.DataFrame(results)],
                    axis=1
                )

                st.subheader("📊 Classification Results")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="classified_emails.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error("Failed to process CSV file")
        st.code(str(e))
