"""
Feature Importance Comparison across three scenarios:
1. Baseline: Standard features
2. Extended: Standard features + new engineered features
3. Replaced: Only new engineered features

All scenarios use identical model parameters for fair comparison.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import load_data from data_loader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data
from data_preprocessing import create_engineered_features


# ── CONSISTENT MODEL PARAMETERS ────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1
}



def train_baseline(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier with consistent parameters
    for fair comparison across scenarios.

    Parameters:
        X: Feature matrix
        y: Target vector

    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def train_and_evaluate_scenario(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
    scenario_name: str
) -> dict:
    """
    Train model for a scenario and return evaluation metrics and feature importance.

    Parameters:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target vectors
        feature_names: List of feature names
        scenario_name: Name of the scenario for reporting

    Returns:
        Dictionary with model metrics and feature importance
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Features: {len(feature_names)}")
    print(f"Feature list: {', '.join(feature_names)}\n")

    # Train model with consistent parameters
    model = train_baseline(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:<35}: {row['importance']:.4f}")

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'feature_importance': importance_df,
        'feature_names': feature_names
    }




def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> None:
    """
    Plot Mean Decrease in Impurity (MDI) based feature importance as a horizontal bar plot,
    sorted in descending order. Save the plot as PNG in outputs/ folder.

    Parameters:
        model: Trained Random Forest model
        feature_names: List of feature names
    """
    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importance_df)), importance_df['importance'], color='#4C9BE8')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel('Feature Importance (MDI)')
    ax.set_title('Random Forest Feature Importance')
    ax.grid(True, axis='x', alpha=0.3)

    # Add importance values as text
    for i, v in enumerate(importance_df['importance']):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    # Create data/Feature Importance Analysis directory if it doesn't exist
    outputs_dir = Path('data/Feature Importance Analysis')
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    output_path = outputs_dir / 'feature_importance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to: {output_path}")


def plot_scenario_comparison(results_baseline, results_extended, results_replaced, new_features):
    """
    Create side-by-side comparison plots and metrics charts for all three scenarios.

    Parameters:
        results_baseline, results_extended, results_replaced: Result dictionaries from train_and_evaluate_scenario
        new_features: List of new feature names to highlight
    """
    output_dir = Path('data/Feature Importance Analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Side-by-side feature importance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (ax, results, scenario_name) in enumerate([
        (axes[0], results_baseline, 'Baseline'),
        (axes[1], results_extended, 'Extended'),
        (axes[2], results_replaced, 'Replaced')
    ]):
        importance_df = results['feature_importance'].head(10)
        colors = ['#FF6B6B' if feat in new_features else '#4C9BE8' 
                  for feat in importance_df['feature']]
        
        ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=10)
        ax.set_title(f'{scenario_name}\n({len(results["feature_names"])} features, F1={results["f1"]:.4f})', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C9BE8', label='Original Features'),
        Patch(facecolor='#FF6B6B', label='New Features')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    fig_path = output_dir / 'feature_importance_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Feature importance comparison plot saved to: {fig_path}")
    plt.close()

    # Plot 2: Metrics comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ['Baseline', 'Extended', 'Replaced']
    x = np.arange(len(scenarios))
    width = 0.2

    metrics = {
        'Accuracy': [results_baseline['accuracy'], results_extended['accuracy'], results_replaced['accuracy']],
        'Precision': [results_baseline['precision'], results_extended['precision'], results_replaced['precision']],
        'Recall': [results_baseline['recall'], results_extended['recall'], results_replaced['recall']],
        'F1-Score': [results_baseline['f1'], results_extended['f1'], results_replaced['f1']]
    }

    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric_name)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for i, (metric_name, values) in enumerate(metrics.items()):
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    comparison_metrics_path = output_dir / 'metrics_comparison.png'
    plt.savefig(comparison_metrics_path, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics comparison plot saved to: {comparison_metrics_path}")
    plt.close()




if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = load_data("data/raw/ai4i2020.csv")

    # Create engineered features
    df = create_engineered_features(df)

    # Define feature sets for each scenario
    baseline_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    new_features = [
        "Temperature difference [K]",
        "Machine power [W]"
    ]

    extended_features = baseline_features + new_features
    
    # Replaced: New features replace specific old ones
    replaced_features = [
        "Temperature difference [K]",     # Replaces: Air temperature + Process temperature
        "Machine power [W]",               # Replaces: Rotational speed + Torque
        "Tool wear [min]"                  # Kept from baseline
    ]

    target_column = "Machine failure"

    # Prepare data
    print("\nPreparing data...")
    y = df[target_column]

    # Train-test split (consistent across all scenarios)
    X_baseline = df[baseline_features]
    X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train_baseline)} samples")
    print(f"Test set: {len(X_test_baseline)} samples")

    # ── SCENARIO 1: BASELINE ────────────────────────────────────────────────────
    results_baseline = train_and_evaluate_scenario(
        X_train_baseline, X_test_baseline,
        y_train, y_test,
        baseline_features,
        "BASELINE (Original Features)"
    )

    # ── SCENARIO 2: EXTENDED (Add new features) ─────────────────────────────────
    X_extended = df[extended_features]
    X_train_extended, X_test_extended, _, _ = train_test_split(
        X_extended, y, test_size=0.2, random_state=42, stratify=y
    )

    results_extended = train_and_evaluate_scenario(
        X_train_extended, X_test_extended,
        y_train, y_test,
        extended_features,
        "EXTENDED (Original + New Features)"
    )

    # ── SCENARIO 3: REPLACED (Only new features) ───────────────────────────────
    X_replaced = df[replaced_features]
    X_train_replaced, X_test_replaced, _, _ = train_test_split(
        X_replaced, y, test_size=0.2, random_state=42, stratify=y
    )

    results_replaced = train_and_evaluate_scenario(
        X_train_replaced, X_test_replaced,
        y_train, y_test,
        replaced_features,
        "REPLACED (Only New Features)"
    )

    # ── CREATE COMPARISON TABLE ────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE: MODEL EVALUATION METRICS")
    print(f"{'='*70}\n")

    comparison_df = pd.DataFrame({
        'Scenario': ['Baseline', 'Extended', 'Replaced'],
        'Features': [
            len(baseline_features),
            len(extended_features),
            len(replaced_features)
        ],
        'Accuracy': [
            results_baseline['accuracy'],
            results_extended['accuracy'],
            results_replaced['accuracy']
        ],
        'Precision': [
            results_baseline['precision'],
            results_extended['precision'],
            results_replaced['precision']
        ],
        'Recall': [
            results_baseline['recall'],
            results_extended['recall'],
            results_replaced['recall']
        ],
        'F1-Score': [
            results_baseline['f1'],
            results_extended['f1'],
            results_replaced['f1']
        ]
    })

    print(comparison_df.to_string(index=False))

    # Calculate improvements
    print(f"\n{'='*70}")
    print("IMPROVEMENTS VS BASELINE")
    print(f"{'='*70}\n")

    print("Extended Scenario:")
    print(f"  Accuracy:  {(results_extended['accuracy'] - results_baseline['accuracy']) / results_baseline['accuracy'] * 100:+.2f}%")
    print(f"  Precision: {(results_extended['precision'] - results_baseline['precision']) / results_baseline['precision'] * 100:+.2f}%")
    print(f"  Recall:    {(results_extended['recall'] - results_baseline['recall']) / results_baseline['recall'] * 100:+.2f}%")
    print(f"  F1-Score:  {(results_extended['f1'] - results_baseline['f1']) / results_baseline['f1'] * 100:+.2f}%")

    print("\nReplaced Scenario:")
    print(f"  Accuracy:  {(results_replaced['accuracy'] - results_baseline['accuracy']) / results_baseline['accuracy'] * 100:+.2f}%")
    print(f"  Precision: {(results_replaced['precision'] - results_baseline['precision']) / results_baseline['precision'] * 100:+.2f}%")
    print(f"  Recall:    {(results_replaced['recall'] - results_baseline['recall']) / results_baseline['recall'] * 100:+.2f}%")
    print(f"  F1-Score:  {(results_replaced['f1'] - results_baseline['f1']) / results_baseline['f1'] * 100:+.2f}%")

    # Create visualization plots
    plot_scenario_comparison(results_baseline, results_extended, results_replaced, new_features)

    # Save detailed feature importance for each scenario
    print(f"\n{'='*70}")
    print("DETAILED FEATURE IMPORTANCE")
    print(f"{'='*70}\n")

    for scenario_name, results in [
        ('Baseline', results_baseline),
        ('Extended', results_extended),
        ('Replaced', results_replaced)
    ]:
        print(f"\n{scenario_name}:")
        print(results['feature_importance'].to_string())

        # Save to CSV
        importance_csv = output_dir / f'feature_importance_{scenario_name.lower()}.csv'
        results['feature_importance'].to_csv(importance_csv, index=False)
        print(f"  → Saved to: {importance_csv}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: {output_dir}")
