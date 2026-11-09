import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

st.title("🎓 Student Performance Prediction System")
st.caption("Classroom prototype for exploring possible academic risk.")

if not Path("student_model.pkl").exists() or not Path("student_data.csv").exists():
    st.error("Required files are missing. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = pd.read_csv("student_data.csv")

c1, c2, c3 = st.columns(3)
c1.metric("Historical records", len(data))
c2.metric("Average final grade (G3)", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Download a sample input file")
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False).encode("utf-8"),
    "sample_prediction_input.csv",
    "text/csv"
)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same input columns used during model training. Do not include G3.")
uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [col for col in required_columns if col not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV has no student records.")
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if value == 1 else "NOT AT RISK"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False).encode("utf-8"),
                "student_predictions.csv",
                "text/csv"
            )
    except Exception as exc:
        st.error(f"Unable to process this file: {exc}")

st.divider()
st.warning(
    "Responsible use: This is a classroom demonstration using historical public data. "
    "Predictions are not diagnoses and must not be the sole basis for decisions about students."
)
