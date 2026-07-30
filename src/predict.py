"""
Prediction module for Iris Flower Classifier.

Provides functions to load the trained model and make predictions
on new flower measurements with confidence scores.
"""

import numpy as np
from src.utils import get_model_path, SPECIES_NAMES
from src.train import load_model


def load_prediction_model(model_path=None):
    """
    Loads the saved model and scaler from saved_model/best_model.pkl.

    Args:
        model_path: Optional custom path to the model file.

    Returns:
        tuple: (model, scaler) — the trained model and its associated scaler.
    """
    loaded = load_model(path=model_path)

    if isinstance(loaded, dict):
        return loaded['model'], loaded.get('scaler', None)
    else:
        return loaded, None


def predict_species(model, features, scaler=None):
    """
    Predicts the species of an Iris flower.

    Args:
        model: Trained scikit-learn model.
        features: List or array of [sepal_length, sepal_width, petal_length, petal_width].
        scaler: Optional StandardScaler to transform features before prediction.

    Returns:
        Predicted species name as a string.
    """
    features_array = np.array(features).reshape(1, -1)
    if scaler is not None:
        features_array = scaler.transform(features_array)
    prediction = model.predict(features_array)[0]
    return prediction


def predict_with_confidence(model, features, scaler=None):
    """
    Predicts species with confidence scores.

    Args:
        model: Trained scikit-learn model.
        features: List or array of [sepal_length, sepal_width, petal_length, petal_width].
        scaler: Optional StandardScaler to transform features before prediction.

    Returns:
        Dict with 'predicted_species', 'confidence', and 'probabilities'.
    """
    features_array = np.array(features).reshape(1, -1)
    if scaler is not None:
        features_array = scaler.transform(features_array)

    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    classes = model.classes_

    prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))

    return {
        'predicted_species': prediction,
        'confidence': confidence,
        'probabilities': prob_dict
    }


def demo_predictions(model, scaler=None):
    """
    Demonstrates predictions on sample inputs with formatted output.

    Args:
        model: Trained scikit-learn model.
        scaler: Optional StandardScaler to transform features.
    """
    samples = [
        {'features': [5.1, 3.5, 1.4, 0.2], 'expected': 'Iris-setosa'},
        {'features': [6.7, 3.1, 4.7, 1.5], 'expected': 'Iris-versicolor'},
        {'features': [6.3, 2.7, 4.9, 1.8], 'expected': 'Iris-virginica'},
    ]

    print("\n  Sample Predictions:")
    print("  " + "-" * 72)
    print(f"  {'Input Features':<30} {'Expected':<18} {'Predicted':<18} {'Conf.'}")
    print("  " + "-" * 72)

    for sample in samples:
        result = predict_with_confidence(model, sample['features'], scaler)
        features_str = str(sample['features'])
        expected = sample['expected']
        predicted = result['predicted_species']
        confidence = result['confidence']
        match = "✓" if expected == predicted else "✗"

        print(f"  {features_str:<30} {expected:<18} {predicted:<18} {confidence:.1%} {match}")

    print("  " + "-" * 72)
