"""
Main Pipeline Script for Iris Flower Classifier.

Runs the complete Machine Learning workflow from data loading
through model training, evaluation, and prediction demo.

Usage:
    python main.py
"""

import os
import sys
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    load_dataset, setup_plot_style, get_output_path,
    FEATURE_COLUMNS, TARGET_COLUMN, SPECIES_NAMES, FEATURE_LABELS
)
from src.preprocess import inspect_dataset, prepare_data
from src.train import (
    train_and_evaluate_all, select_best_model, save_model
)
from src.predict import demo_predictions


def print_step(step_num, title):
    """Prints a formatted step header."""
    print('\n' + '=' * 70)
    print(f'  STEP {step_num}: {title.upper()}')
    print('=' * 70)


def generate_pairplot(df, palette):
    """Generates and saves the pair plot."""
    print("  Generating pairplot...")
    g = sns.pairplot(df, hue=TARGET_COLUMN, palette=palette, height=2.5,
                     plot_kws={'alpha': 0.7, 'edgecolor': 'white', 'linewidth': 0.5})
    g.figure.suptitle('Pairplot of Iris Features by Species', y=1.02, fontsize=14, fontweight='bold')
    save_path = get_output_path('pairplot.png')
    g.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {save_path}")


def generate_heatmap(df):
    """Generates and saves the correlation heatmap."""
    print("  Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_COLUMNS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                square=True, fmt='.2f', linewidths=1, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap of Iris Features', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(FEATURE_LABELS, rotation=0)
    fig.tight_layout()
    save_path = get_output_path('heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_histograms(df, palette):
    """Generates and saves histograms of each feature."""
    print("  Generating histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (feature, label) in enumerate(zip(FEATURE_COLUMNS, FEATURE_LABELS)):
        sns.histplot(data=df, x=feature, hue=TARGET_COLUMN,
                     palette=palette, kde=True, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {label}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Count')
    fig.suptitle('Feature Distributions by Species', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_path = get_output_path('histogram.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_boxplots(df, palette):
    """Generates and saves box plots of each feature."""
    print("  Generating boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (feature, label) in enumerate(zip(FEATURE_COLUMNS, FEATURE_LABELS)):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=feature,
                    palette=palette, ax=axes[i], width=0.6)
        axes[i].set_title(f'{label} by Species', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Species')
        axes[i].set_ylabel(label)
    fig.suptitle('Feature Box Plots by Species', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_path = get_output_path('boxplot.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_confusion_matrix_plot(conf_matrix, model_name):
    """Generates and saves the confusion matrix visualization."""
    print("  Generating confusion matrix plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=SPECIES_NAMES, yticklabels=SPECIES_NAMES,
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    fig.tight_layout()
    save_path = get_output_path('confusion_matrix.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_model_comparison_chart(results):
    """Generates and saves the model accuracy comparison bar chart."""
    print("  Generating model comparison chart...")
    model_names = list(results.keys())
    accuracies = [results[name]['metrics']['accuracy'] for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(model_names, accuracies, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    save_path = get_output_path('model_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_comparison_table(results):
    """Prints a formatted model comparison table."""
    header = (
        "╔" + "═" * 22 + "╦" + "═" * 10 + "╦" + "═" * 11 +
        "╦" + "═" * 8 + "╦" + "═" * 10 + "╗"
    )
    row_sep = (
        "╠" + "═" * 22 + "╬" + "═" * 10 + "╬" + "═" * 11 +
        "╬" + "═" * 8 + "╬" + "═" * 10 + "╣"
    )
    footer = (
        "╚" + "═" * 22 + "╩" + "═" * 10 + "╩" + "═" * 11 +
        "╩" + "═" * 8 + "╩" + "═" * 10 + "╝"
    )

    print(header)
    print("║ {:<20} ║ {:>8} ║ {:>9} ║ {:>6} ║ {:>8} ║".format(
        "Model", "Accuracy", "Precision", "Recall", "F1 Score"))
    print(row_sep)

    for name, data in results.items():
        m = data['metrics']
        print("║ {:<20} ║ {:>8.4f} ║ {:>9.4f} ║ {:>6.4f} ║ {:>8.4f} ║".format(
            name, m['accuracy'], m['precision'], m['recall'], m['f1_score']))

    print(footer)


def main():
    """Runs the complete Iris Flower Classification pipeline."""

    print("\n" + "█" * 70)
    print("  🌸 IRIS FLOWER CLASSIFIER — ML PIPELINE")
    print("  CodeAlpha Data Science Internship")
    print("█" * 70)

    # Setup visualization style
    setup_plot_style()
    palette = ['#2ecc71', '#3498db', '#e74c3c']

    # ── Step 1: Load Dataset ──────────────────────────────────────────────
    print_step(1, "Loading Dataset")
    df = load_dataset()
    print(f"  Dataset loaded successfully.")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Step 2: Inspect Dataset ───────────────────────────────────────────
    print_step(2, "Inspecting Dataset")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\n  Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<20} {dtype}")

    # ── Step 3: Check Missing Values ──────────────────────────────────────
    print_step(3, "Checking Missing Values")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("  ✓ No missing values found.")
    else:
        print(f"  ✗ Found {total_missing} missing values:")
        print(missing[missing > 0].to_string())

    # ── Step 4: Check Duplicates ──────────────────────────────────────────
    print_step(4, "Checking Duplicate Values")
    duplicates = df.duplicated().sum()
    if duplicates == 0:
        print("  ✓ No duplicate rows found.")
    else:
        print(f"  Found {duplicates} duplicate row(s).")

    # ── Step 5: Summary Statistics ────────────────────────────────────────
    print_step(5, "Summary Statistics")
    print(df.describe().to_string())

    # ── Step 6: Exploratory Data Analysis ─────────────────────────────────
    print_step(6, "Exploratory Data Analysis")
    print("  Class Distribution:")
    for species, count in df[TARGET_COLUMN].value_counts().items():
        print(f"    {species}: {count} samples")

    generate_pairplot(df, palette)
    generate_heatmap(df)
    generate_histograms(df, palette)
    generate_boxplots(df, palette)
    print("  ✓ All EDA plots saved to outputs/graphs/")

    # ── Step 7: Data Preprocessing ────────────────────────────────────────
    print_step(7, "Data Preprocessing")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Features scaled with StandardScaler.")

    # ── Step 8: Train Models ──────────────────────────────────────────────
    print_step(8, "Training Models")
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    for name in results:
        acc = results[name]['metrics']['accuracy']
        print(f"  ✓ {name}: Accuracy = {acc:.4f}")

    # ── Step 9: Compare Models ────────────────────────────────────────────
    print_step(9, "Model Comparison")
    print_comparison_table(results)
    generate_model_comparison_chart(results)

    # ── Step 10: Select Best Model ────────────────────────────────────────
    print_step(10, "Selecting Best Model")
    best_name, best_model, best_metrics = select_best_model(results)
    print(f"  ★ Best Model: {best_name}")
    print(f"    Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"    Precision: {best_metrics['precision']:.4f}")
    print(f"    Recall:    {best_metrics['recall']:.4f}")
    print(f"    F1 Score:  {best_metrics['f1_score']:.4f}")

    # ── Step 11: Evaluate Final Model ─────────────────────────────────────
    print_step(11, "Evaluating Final Model")
    print(f"\n  Classification Report for {best_name}:\n")
    print(best_metrics['classification_report'])
    print(f"  Confusion Matrix:")
    print(best_metrics['confusion_matrix'])
    generate_confusion_matrix_plot(best_metrics['confusion_matrix'], best_name)

    # ── Step 12: Save Model ───────────────────────────────────────────────
    print_step(12, "Saving Model")
    model_path = save_model(best_model, scaler=scaler)
    print(f"  ✓ Model saved to: {model_path}")

    # ── Step 13: Prediction Demo ──────────────────────────────────────────
    print_step(13, "Prediction Demo")
    demo_predictions(best_model, scaler)

    # ── Final Summary ─────────────────────────────────────────────────────
    print('\n' + '█' * 70)
    print('  ✅ PIPELINE COMPLETE')
    print('█' * 70)
    print(f"  Best Model : {best_name}")
    print(f"  Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"  Model Path : {model_path}")
    print(f"  Graphs     : outputs/graphs/")
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
