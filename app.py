"""
Streamlit web application for Iris Flower Classification.

An interactive dashboard for predicting Iris flower species
using a trained Machine Learning model.

Usage:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_model", "best_model.pkl")

# Constants
SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
SPECIES_INFO = {
    "Iris-setosa": {"icon": ":material/eco:", "color": "green", "label": "Setosa"},
    "Iris-versicolor": {"icon": ":material/local_florist:", "color": "blue", "label": "Versicolor"},
    "Iris-virginica": {"icon": ":material/park:", "color": "orange", "label": "Virginica"},
}

# Page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon=":material/local_florist:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    """Loads the trained model and scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict):
            return loaded["model"], loaded.get("scaler", None)
        return loaded, None
    except Exception as e:
        st.error(f"Error loading model: {e}", icon=":material/error:")
        return None, None


model, scaler = load_trained_model()

if model is None:
    st.error(
        "Model not found. Run `python main.py` first to train and save the model.",
        icon=":material/error:",
    )
    st.stop()


# ── Sidebar — informational content only ──────────────────────────────────
with st.sidebar:
    st.header("About", divider="gray")

    with st.expander("About the model", icon=":material/model_training:"):
        st.markdown(
            """
            This app uses a **Decision Tree Classifier** trained on the Iris dataset.
            Three models were compared during training:

            | Model | Type |
            |-------|------|
            | Decision Tree | Tree-based |
            | K-Nearest Neighbors | Instance-based |
            | Logistic Regression | Linear |

            The best-performing model was automatically selected based on accuracy,
            precision, recall, and F1 score.
            """
        )

    with st.expander("About the dataset", icon=":material/dataset:"):
        st.markdown(
            """
            The **Iris dataset** was introduced by Ronald Fisher in 1936.

            - **150 samples** (50 per species)
            - **3 species** — Setosa, Versicolor, Virginica
            - **4 features** — Sepal length & width, Petal length & width

            Source: UCI Machine Learning Repository
            """
        )

    with st.expander("How prediction works", icon=":material/psychology:"):
        st.markdown(
            """
            1. Enter the four flower measurements
            2. Features are scaled using StandardScaler
            3. The trained model predicts the species
            4. Prediction probabilities show confidence for each class
            """
        )

    with st.expander("Technologies used", icon=":material/code:"):
        st.markdown(
            """
            - **Python** — Core language
            - **Scikit-learn** — ML model training
            - **Pandas & NumPy** — Data processing
            - **Streamlit** — Web interface
            - **Joblib** — Model serialization
            """
        )

    st.caption(
        "Developed by **Sundeep Patil**  \nCodeAlpha Data Science Internship"
    )


# ── Main content ──────────────────────────────────────────────────────────
st.title("Iris flower species classifier")
st.caption("Machine learning classification using Scikit-Learn")

# ── Input features card ──────────────────────────────────────────────────
st.subheader("Input features", divider="gray")

col_sepal, col_petal = st.columns(2)

with col_sepal:
    with st.container(border=True):
        st.markdown(":material/straighten: **Sepal measurements**")
        sepal_length = st.slider(
            "Sepal length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.4,
            step=0.1,
        )
        sepal_width = st.slider(
            "Sepal width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.4,
            step=0.1,
        )

with col_petal:
    with st.container(border=True):
        st.markdown(":material/straighten: **Petal measurements**")
        petal_length = st.slider(
            "Petal length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.3,
            step=0.1,
        )
        petal_width = st.slider(
            "Petal width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=0.2,
            step=0.1,
        )

# Predict button
predict_btn = st.button(
    "Predict species",
    type="primary",
    icon=":material/search:",
    use_container_width=True,
)

# ── Prediction ────────────────────────────────────────────────────────────
if predict_btn:
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Map prediction to species info
    predicted_species = str(prediction)
    classes = list(model.classes_)
    idx = classes.index(prediction) if prediction in classes else 0
    confidence = probabilities[idx] * 100
    info = SPECIES_INFO.get(predicted_species, SPECIES_INFO["Iris-setosa"])

    # ── Results ───────────────────────────────────────────────────────
    st.subheader("Prediction results", divider="gray")

    col_result, col_confidence, col_summary = st.columns(3)

    with col_result:
        with st.container(border=True):
            st.metric(
                label="Predicted species",
                value=info["label"],
            )
            st.badge(predicted_species, icon=info["icon"], color=info["color"])

    with col_confidence:
        with st.container(border=True):
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%",
            )
            st.progress(min(int(confidence), 100))

    with col_summary:
        with st.container(border=True):
            st.markdown(":material/summarize: **Input summary**")
            summary_df = pd.DataFrame(
                {
                    "Feature": [
                        "Sepal length",
                        "Sepal width",
                        "Petal length",
                        "Petal width",
                    ],
                    "Value (cm)": [
                        sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width,
                    ],
                }
            ).set_index("Feature")
            st.dataframe(summary_df, use_container_width=True)

    # ── Probability chart ─────────────────────────────────────────────
    st.subheader("Prediction probabilities", divider="gray")

    prob_df = pd.DataFrame(
        {
            "Species": [str(c) for c in classes],
            "Probability (%)": [round(float(p) * 100, 1) for p in probabilities],
        }
    ).set_index("Species")

    st.bar_chart(prob_df, horizontal=True, use_container_width=True)

else:
    # Default state
    st.info(
        "Adjust the sliders above and click **Predict species** to classify an Iris flower.",
        icon=":material/touch_app:",
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.caption(
    "Developed using Streamlit · Scikit-Learn | CodeAlpha Data Science Internship Project | Developed by Sundeep Patil",
    help="https://github.com/sundeep-codes/CodeAlpha_Iris_Flower_Classifier",
)
