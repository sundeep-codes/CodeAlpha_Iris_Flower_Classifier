<div align="center">

# 🌸 Iris Flower Species Classifier

### Machine Learning Classification using Scikit-Learn

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

A production-quality ML classification project that predicts Iris flower species from petal and sepal measurements. Includes a complete data science pipeline, model comparison, and an interactive Streamlit dashboard.

**CodeAlpha Data Science Internship Project**

[Features](#-features) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results) · [License](#-license)

</div>

---

## 📋 Project overview

**Problem statement** — Given four physical measurements of an Iris flower (sepal length, sepal width, petal length, petal width), predict which of three species it belongs to: *Setosa*, *Versicolor*, or *Virginica*.

**Solution** — A complete machine learning pipeline that loads data, performs exploratory data analysis, trains three classification models, selects the best performer, and deploys it through an interactive web application.

**ML workflow:**

```
Load Data → Inspect & Clean → EDA Visualizations → Feature Scaling
    → Train 3 Models → Compare Metrics → Select Best → Save Model
        → Deploy Streamlit Dashboard → Real-time Predictions
```

Built as part of the **CodeAlpha Data Science Internship**, this project demonstrates professional data science practices including modular code architecture, reproducible pipelines, and interactive model deployment.

---

## 🎯 Objectives

- Perform comprehensive exploratory data analysis with professional visualizations
- Train and compare three classification algorithms on the Iris dataset
- Select the best-performing model using accuracy, precision, recall, and F1 score
- Deploy an interactive prediction dashboard with real-time confidence scores
- Maintain production-quality, modular, PEP8-compliant code

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Complete EDA pipeline** | Pair plots, correlation heatmaps, histograms, box plots |
| **3 ML models** | Decision Tree, K-Nearest Neighbors, Logistic Regression |
| **Automated model selection** | Best model chosen by accuracy with full metric comparison |
| **Interactive dashboard** | Dark-themed Streamlit UI with sliders and live predictions |
| **Prediction confidence** | Probability scores for each species with visual charts |
| **Modular codebase** | Separate modules for utils, preprocessing, training, prediction |

---

## 📊 Dataset

The classic **Iris dataset** introduced by Ronald Fisher (1936), sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

| Feature | Description | Range |
|---------|-------------|-------|
| Sepal Length | Length of the sepal (cm) | 4.3 – 7.9 |
| Sepal Width | Width of the sepal (cm) | 2.0 – 4.4 |
| Petal Length | Length of the petal (cm) | 1.0 – 6.9 |
| Petal Width | Width of the petal (cm) | 0.1 – 2.5 |

**Target classes:** Iris-setosa · Iris-versicolor · Iris-virginica (50 samples each, 150 total)

---

## 🛠️ Technology stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Static visualizations |
| Seaborn | Statistical plotting |
| Scikit-learn | Model training and evaluation |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

---

## 📁 Project structure

```
CodeAlpha_Iris_Flower_Classifier/
│
├── data/
│   └── Iris.csv                           # Dataset (150 samples)
│
├── notebooks/
│   └── Iris_Flower_Classification.ipynb   # Jupyter notebook walkthrough
│
├── src/
│   ├── __init__.py                        # Package init
│   ├── utils.py                           # Path helpers, constants, plot config
│   ├── preprocess.py                      # Data inspection & train/test split
│   ├── train.py                           # Model training, evaluation, save/load
│   └── predict.py                         # Prediction with confidence scores
│
├── saved_model/
│   └── best_model.pkl                     # Trained model + scaler
│
├── outputs/
│   └── graphs/                            # Generated EDA visualizations
│       ├── pairplot.png
│       ├── heatmap.png
│       ├── histogram.png
│       ├── boxplot.png
│       ├── confusion_matrix.png
│       └── model_comparison.png
│
├── screenshots/                           # App screenshots
├── .streamlit/
│   └── config.toml                        # Dark theme configuration
├── app.py                                 # Streamlit dashboard
├── main.py                                # ML pipeline entry point
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
└── .gitignore                             # Git ignore rules
```

---

## 🧪 Machine learning workflow

### Models used

| # | Model | Key characteristic |
|---|-------|-------------------|
| 1 | **Decision Tree** | Splits data using feature thresholds; highly interpretable |
| 2 | **K-Nearest Neighbors** | Classifies by majority vote of 5 nearest neighbors |
| 3 | **Logistic Regression** | Linear decision boundary with softmax probabilities |

### Evaluation metrics

Each model is evaluated using:
- **Accuracy** — Overall correct predictions
- **Precision** — Correct positive predictions per class (weighted)
- **Recall** — True positive rate per class (weighted)
- **F1 Score** — Harmonic mean of precision and recall (weighted)

---

## 📈 Results

### Model performance comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% |
| K-Nearest Neighbors | 100.00% | 100.00% | 100.00% | 100.00% |
| Logistic Regression | 100.00% | 100.00% | 100.00% | 100.00% |

> *Results from the last pipeline run with `random_state=42` and 80/20 train-test split. Values may vary slightly with different random states.*

### Key findings

- **Iris-setosa** is linearly separable from the other two classes
- **Petal measurements** are more discriminative than sepal measurements
- All three models achieve high accuracy, reflecting the dataset's clean separation
- The best model is automatically selected and saved for deployment

---

## 📸 Screenshots

*Run `streamlit run app.py` and add screenshots to the `screenshots/` folder.*

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/sundeep-codes/CodeAlpha_Iris_Flower_Classifier.git
cd CodeAlpha_Iris_Flower_Classifier

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the complete ML pipeline

```bash
python main.py
```

This executes all 13 steps: data loading, inspection, EDA, preprocessing, model training, comparison, evaluation, model saving, and prediction demo.

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Opens an interactive web app where you can adjust flower measurements and get real-time species predictions with confidence scores.

### Make predictions programmatically

```python
from src.predict import load_prediction_model, predict_with_confidence

model, scaler = load_prediction_model()
result = predict_with_confidence(model, [5.1, 3.5, 1.4, 0.2], scaler)

print(f"Species: {result['predicted_species']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## 🔮 Future improvements

- Add k-fold cross-validation for robust model evaluation
- Implement feature importance analysis and visualization
- Deploy the application to Streamlit Cloud
- Add REST API endpoint using FastAPI
- Implement model versioning with MLflow
- Add unit tests for the prediction pipeline

---

## 👨‍💻 Author

**Sundeep Patil**

CodeAlpha Data Science Intern

[![GitHub](https://img.shields.io/badge/GitHub-sundeep--codes-181717?style=flat-square&logo=github)](https://github.com/sundeep-codes)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ Star this repository if you found it useful!

*Built for the CodeAlpha Data Science Internship*

</div>
