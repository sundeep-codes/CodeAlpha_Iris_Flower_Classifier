"""
Iris Flower Species Classifier — ML Dashboard
Premium Streamlit application for the CodeAlpha Data Science Internship.

Usage:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

# ── Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_model", "best_model.pkl")

SPECIES = {
    "Iris-setosa": {
        "label": "Setosa",
        "scientific": "Iris setosa",
        "icon": "🌿",
        "color": "#22c55e",
        "desc": "The smallest species with short, broad petals and bristle-tipped sepals.",
        "habitat": "Arctic and subarctic regions of North America and East Asia.",
        "fact": "Easily separable from other species due to distinctly smaller petal dimensions.",
    },
    "Iris-versicolor": {
        "label": "Versicolor",
        "scientific": "Iris versicolor",
        "icon": "🌺",
        "color": "#6366f1",
        "desc": "Medium-sized iris with violet-blue flowers, known as the Blue Flag.",
        "habitat": "Marshes, wet meadows, and shorelines across eastern North America.",
        "fact": "Its root was traditionally used in medicine by Native American communities.",
    },
    "Iris-virginica": {
        "label": "Virginica",
        "scientific": "Iris virginica",
        "icon": "🌻",
        "color": "#f97316",
        "desc": "The largest species with broad sepals and prominent upright petals.",
        "habitat": "Coastal plains and wetlands of the southeastern United States.",
        "fact": "Closely resembles Versicolor but distinguished by larger flower dimensions.",
    },
}

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier · ML Dashboard",
    page_icon=":material/local_florist:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Accent CSS (glassmorphism, gradient header, prediction card) ──────────
st.markdown(
    """<style>
.hero-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fafafa 0%, #a1a1aa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 4px; line-height: 1.2;
}
.hero-sub { font-size: 1rem; color: #71717a; margin: 0 0 12px; }
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.hbadge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 12px; border-radius: 20px; font-size: 0.76rem; font-weight: 500;
    border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04);
}
.hbadge-g { border-color: rgba(34,197,94,0.3); color: #4ade80; }
.hbadge-i { border-color: rgba(99,102,241,0.3); color: #a5b4fc; }
.hbadge-a { border-color: rgba(245,158,11,0.3); color: #fbbf24; }
.pred-card {
    background: rgba(255,255,255,0.03); backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 20px;
    padding: 28px; text-align: center;
}
.pred-icon { font-size: 3rem; }
.pred-name { font-size: 1.5rem; font-weight: 700; margin: 4px 0 0; }
.pred-sci { font-size: 0.82rem; color: #71717a; font-style: italic; margin: 2px 0 16px; }
.pred-clbl {
    font-size: 0.7rem; color: #71717a; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600;
}
.pred-cval { font-size: 2.5rem; font-weight: 800; line-height: 1.1; margin: 4px 0 8px; }
.pred-bar {
    height: 5px; border-radius: 3px; background: rgba(255,255,255,0.06); overflow: hidden;
}
.pred-fill { height: 100%; border-radius: 3px; }
.info-block {
    text-align: left; margin-top: 20px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.info-item { margin-bottom: 12px; }
.info-lbl {
    font-size: 0.68rem; color: #71717a; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; margin-bottom: 2px;
}
.info-val { font-size: 0.84rem; color: #d4d4d8; line-height: 1.5; }
.app-footer {
    text-align: center; padding: 28px 0 8px;
    border-top: 1px solid rgba(255,255,255,0.06); margin-top: 32px;
}
.app-footer p { font-size: 0.8rem; color: #52525b; margin: 3px 0; }
.app-footer a { color: #818cf8; text-decoration: none; }
</style>""",
    unsafe_allow_html=True,
)


# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load trained model and scaler from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            return data["model"], data.get("scaler")
        return data, None
    except Exception:
        return None, None


model, scaler = load_model()
if model is None:
    st.error(
        "Model not found. Run `python main.py` first to train the model.",
        icon=":material/error:",
    )
    st.stop()


# ── Sidebar — Information center ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### :material/info: Information center")

    with st.expander("About the model", icon=":material/model_training:"):
        st.markdown("""
        Uses a **Decision Tree Classifier** selected from three candidates:
        - Decision Tree Classifier
        - K-Nearest Neighbors (K=5)
        - Logistic Regression (max_iter=200)

        Selection based on accuracy, precision, recall, and F1 score.
        Features scaled with StandardScaler before training.
        """)

    with st.expander("Dataset details", icon=":material/dataset:"):
        st.markdown("""
        **Iris dataset** — Ronald Fisher, 1936

        | Attribute | Value |
        |-----------|-------|
        | Samples | 150 |
        | Features | 4 |
        | Classes | 3 |
        | Balance | 50 per class |

        Source: UCI Machine Learning Repository
        """)

    with st.expander("ML workflow", icon=":material/account_tree:"):
        st.markdown("""
        1. Load & inspect dataset
        2. Exploratory data analysis
        3. Feature scaling (StandardScaler)
        4. Train three classifiers
        5. Evaluate & compare metrics
        6. Select best model
        7. Save model + scaler
        8. Deploy dashboard
        """)

    with st.expander("Technologies", icon=":material/code:"):
        st.markdown("""
        **Core** — Python · Scikit-learn · Joblib

        **Data** — Pandas · NumPy

        **Visualization** — Matplotlib · Seaborn · Plotly

        **Deployment** — Streamlit
        """)

    with st.expander("Developer", icon=":material/person:"):
        st.markdown("""
        **Sundeep Patil**
        CodeAlpha Data Science Intern

        [:material/link: GitHub](https://github.com/sundeep-codes)
        · [:material/link: Repository](https://github.com/sundeep-codes/CodeAlpha_Iris_Flower_Classifier)
        """)


# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
<h1 class="hero-title">Iris Flower Species Classifier</h1>
<p class="hero-sub">Machine learning classification powered by Scikit-Learn</p>
<div class="badge-row">
    <span class="hbadge hbadge-g">● 100% Accuracy</span>
    <span class="hbadge hbadge-i">Decision Tree</span>
    <span class="hbadge hbadge-a">150 Samples</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("")


# ── Metrics row ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model accuracy", "100%")
m2.metric("Dataset size", "150 samples")
m3.metric("Species count", "3 classes")
m4.metric("Best model", "Decision Tree")

st.markdown("")


# ── Feature inputs + Prediction ───────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("Feature measurements", divider="gray")

    sl_col, sw_col = st.columns(2)
    with sl_col:
        with st.container(border=True):
            st.markdown(":material/straighten: **Sepal length**")
            st.caption("Length of the outer sepal in centimeters")
            sepal_length = st.slider(
                "Sepal length", 4.0, 8.0, 5.4, 0.1, label_visibility="collapsed"
            )
            st.caption(f"**{sepal_length} cm** · Range: 4.0 – 8.0")

    with sw_col:
        with st.container(border=True):
            st.markdown(":material/straighten: **Sepal width**")
            st.caption("Width of the outer sepal in centimeters")
            sepal_width = st.slider(
                "Sepal width", 2.0, 4.5, 3.4, 0.1, label_visibility="collapsed"
            )
            st.caption(f"**{sepal_width} cm** · Range: 2.0 – 4.5")

    pl_col, pw_col = st.columns(2)
    with pl_col:
        with st.container(border=True):
            st.markdown(":material/straighten: **Petal length**")
            st.caption("Length of the inner petal in centimeters")
            petal_length = st.slider(
                "Petal length", 1.0, 7.0, 1.3, 0.1, label_visibility="collapsed"
            )
            st.caption(f"**{petal_length} cm** · Range: 1.0 – 7.0")

    with pw_col:
        with st.container(border=True):
            st.markdown(":material/straighten: **Petal width**")
            st.caption("Width of the inner petal in centimeters")
            petal_width = st.slider(
                "Petal width", 0.1, 2.5, 0.2, 0.1, label_visibility="collapsed"
            )
            st.caption(f"**{petal_width} cm** · Range: 0.1 – 2.5")

    # Input summary
    st.markdown("")
    st.caption(":material/summarize: **Input summary**")
    is1, is2, is3, is4 = st.columns(4)
    is1.metric("Sepal L", f"{sepal_length}")
    is2.metric("Sepal W", f"{sepal_width}")
    is3.metric("Petal L", f"{petal_length}")
    is4.metric("Petal W", f"{petal_width}")


with right_col:
    st.subheader("Prediction", divider="gray")

    # Run prediction
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    )
    if scaler:
        features_scaled_arr = scaler.transform(features)
        features_scaled = pd.DataFrame(features_scaled_arr, columns=features.columns)
    else:
        features_scaled = features

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    predicted = str(prediction)
    classes = list(model.classes_)
    idx = classes.index(prediction) if prediction in classes else 0
    confidence = probabilities[idx] * 100
    sp = SPECIES.get(predicted, SPECIES["Iris-setosa"])

    # Prediction card (glassmorphism)
    st.markdown(
        f"""
    <div class="pred-card">
        <div class="pred-icon">{sp['icon']}</div>
        <div class="pred-name" style="color:{sp['color']}">{sp['label']}</div>
        <div class="pred-sci">{sp['scientific']}</div>
        <div class="pred-clbl">Confidence</div>
        <div class="pred-cval" style="color:{sp['color']}">{confidence:.1f}%</div>
        <div class="pred-bar">
            <div class="pred-fill" style="width:{confidence}%;background:{sp['color']}"></div>
        </div>
        <div class="info-block">
            <div class="info-item">
                <div class="info-lbl">Description</div>
                <div class="info-val">{sp['desc']}</div>
            </div>
            <div class="info-item">
                <div class="info-lbl">Habitat</div>
                <div class="info-val">{sp['habitat']}</div>
            </div>
            <div class="info-item">
                <div class="info-lbl">Interesting fact</div>
                <div class="info-val">{sp['fact']}</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Probability chart ─────────────────────────────────────────────────────
st.markdown("")
st.subheader("Prediction probabilities", divider="gray")

chart_colors = [SPECIES[c]["color"] if c in SPECIES else "#6366f1" for c in classes]
chart_labels = [SPECIES[c]["label"] if c in SPECIES else c for c in classes]
probs_pct = [round(float(p) * 100, 1) for p in probabilities]

fig = go.Figure(
    go.Bar(
        x=probs_pct,
        y=chart_labels,
        orientation="h",
        marker_color=chart_colors,
        text=[f"{p}%" for p in probs_pct],
        textposition="auto",
        textfont=dict(size=13, family="Inter", color="#fafafa"),
    )
)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a1a1aa", size=12),
    xaxis=dict(
        title="Probability (%)",
        range=[0, 105],
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=False,
    ),
    yaxis=dict(title="", autorange="reversed"),
    margin=dict(l=0, r=0, t=8, b=0),
    height=180,
    bargap=0.35,
)
st.plotly_chart(fig, width="stretch")


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-footer">
    <p>Developed using <strong>Streamlit</strong> · <strong>Scikit-Learn</strong></p>
    <p>CodeAlpha Data Science Internship Project</p>
    <p>Designed & Developed by <strong>Sundeep Patil</strong></p>
    <p><a href="https://github.com/sundeep-codes/CodeAlpha_Iris_Flower_Classifier"
          target="_blank">GitHub Repository</a></p>
</div>
""",
    unsafe_allow_html=True,
)
