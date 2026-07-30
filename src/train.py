"""
Model training and evaluation module for Iris Flower Classifier.

Provides functions to train Decision Tree, K-Nearest Neighbors,
and Logistic Regression classifiers, evaluate their performance,
and save/load the best model.
"""

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from src.utils import get_model_path


def get_models():
    """
    Returns a dictionary of model names mapped to model instances.

    Returns:
        dict: {'Model Name': sklearn_model_instance}
    """
    return {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
    }


def train_model(model, X_train, y_train):
    """
    Trains a model on the training data.

    Args:
        model: Scikit-learn model instance.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data.

    Args:
        model: Trained scikit-learn model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall,
              f1_score, classification_report, confusion_matrix, predictions.
    """
    predictions = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, predictions),
        'predictions': predictions
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates all three models.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        dict: Results for each model with 'model' and 'metrics' keys.
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        trained_model = train_model(model, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)
        results[name] = {
            'model': trained_model,
            'metrics': metrics
        }

    return results


def select_best_model(results):
    """
    Selects the model with the highest accuracy.

    Args:
        results: Dict of model results from train_and_evaluate_all.

    Returns:
        tuple: (model_name, model_instance, metrics_dict)
    """
    best_model_name = None
    best_model = None
    best_metrics = None
    best_accuracy = -1.0

    for name, data in results.items():
        accuracy = data['metrics']['accuracy']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = data['model']
            best_metrics = data['metrics']

    return best_model_name, best_model, best_metrics


def save_model(model, path=None, scaler=None):
    """
    Saves the model (and optionally scaler) using joblib.

    If a scaler is provided, saves a dict with both model and scaler.
    Otherwise, saves just the model.

    Args:
        model: Trained scikit-learn model.
        path: Optional custom save path.
        scaler: Optional StandardScaler instance.

    Returns:
        str: Path where the model was saved.
    """
    if path is None:
        path = get_model_path()

    save_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(save_data, path)
    return path


def load_model(path=None):
    """
    Loads the saved model from disk.

    Returns the model directly if saved as a plain model,
    or a dict with 'model' and 'scaler' if saved with a scaler.

    Args:
        path: Optional custom path to the model file.

    Returns:
        Loaded model or dict with model and scaler.
    """
    if path is None:
        path = get_model_path()
    return joblib.load(path)
