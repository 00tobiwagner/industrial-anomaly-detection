"""
Random Forest Model Comparison
Compares baseline Random Forest with class-weighted variant.
Includes classification report and ROC curve analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Import functions from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data
from data_preprocessing import create_engineered_features


def train_and_evaluate_rf(
    X_train, X_test, y_train, y_test,
    model_name: str,
    class_weight=None
) -> dict:
    """
    Train Random Forest and evaluate with classification report and ROC curve.

    Parameters:
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test target vectors
        model_name: Name of the model for reporting
        class_weight: If 'balanced', adjusts weights inversely proportional to class frequency

    Returns:
        Dictionary with model, predictions, and metrics
    """
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}\n")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,              # Anzahl der Bäume im Wald (mehr = bessere Genauigkeit, aber langsamer)
        max_depth=10,                  # Maximale Tiefe jedes Baumes (verhindert Overfitting)
        min_samples_split=10,          # Mindestanzahl Samples zum Teilen eines Knotens (mehr = einfachere Bäume)
        min_samples_leaf=5,            # Mindestanzahl Samples in einem Blattknoten (mehr = einfachere Bäume)
        random_state=42,               # Seed für Reproduzierbarkeit
        n_jobs=-1,                     # Nutze alle verfügbaren CPU-Kerne für Parallelisierung
        class_weight=class_weight      # Gewichtung der Klassen: None (gleich) oder 'balanced' (inversional zu Häufigkeit)
    )

    model.fit(X_train, y_train)
    print(f"✓ Model trained on {len(X_train)} samples")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Classification Report
    print(f"\nClassification Report (by class):")
    print("-" * 70)
    report = classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'], digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

    # ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {ap_score:.4f}")

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']:<35}: {row['importance']:.4f}")

    return {
        'model': model,
        'model_name': model_name,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'ap_score': ap_score,
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }


def plot_roc_curves(results_baseline, results_balanced):
    """
    Plot ROC curves for both models side-by-side.

    Parameters:
        results_baseline: Results dictionary from baseline model
        results_balanced: Results dictionary from balanced model
    """
    output_dir = Path('data/Feature Importance Analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ROC curves side-by-side
    for ax, results in [(ax1, results_baseline), (ax2, results_balanced)]:
        ax.plot(results['fpr'], results['tpr'],
                label=f"ROC ({results['model_name']}, AUC={results['roc_auc']:.4f})",
                linewidth=2.5, color='#4C9BE8')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f"{results['model_name']}", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    roc_path = output_dir / 'roc_curves_random_forest.png'
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ ROC curve plot saved to: {roc_path}")
    plt.close()

    # Plot 2: Comparison of ROC curves in one plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(results_baseline['fpr'], results_baseline['tpr'],
            label=f"Baseline (AUC={results_baseline['roc_auc']:.4f})",
            linewidth=2.5, color='#4C9BE8')
    ax.plot(results_balanced['fpr'], results_balanced['tpr'],
            label=f"Balanced Weights (AUC={results_balanced['roc_auc']:.4f})",
            linewidth=2.5, color='#FF6B6B')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison: Random Forest Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    comparison_path = output_dir / 'roc_curves_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC comparison plot saved to: {comparison_path}")
    plt.close()


def plot_precision_recall_curves(results_baseline, results_balanced):
    """
    Plot Precision-Recall curves for both models.

    Parameters:
        results_baseline: Results dictionary from baseline model
        results_balanced: Results dictionary from balanced model
    """
    output_dir = Path('data/Feature Importance Analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: PR curves side-by-side
    for ax, results in [(ax1, results_baseline), (ax2, results_balanced)]:
        ax.plot(results['recall'], results['precision'],
                label=f"PR ({results['model_name']}, AP={results['ap_score']:.4f})",
                linewidth=2.5, color='#4C9BE8')
        ax.axhline(y=(results['y_pred'] == 1).sum() / len(results['y_pred']),
                   color='k', linestyle='--', linewidth=1, label='Baseline (No Skill)')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f"{results['model_name']}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    pr_path = output_dir / 'precision_recall_curves_random_forest.png'
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Precision-Recall curve plot saved to: {pr_path}")
    plt.close()

    # Plot 2: Comparison of PR curves in one plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(results_baseline['recall'], results_baseline['precision'],
            label=f"Baseline (AP={results_baseline['ap_score']:.4f})",
            linewidth=2.5, color='#4C9BE8')
    ax.plot(results_balanced['recall'], results_balanced['precision'],
            label=f"Balanced Weights (AP={results_balanced['ap_score']:.4f})",
            linewidth=2.5, color='#FF6B6B')
    
    # Baseline (no skill) - horizontal line at the proportion of positive class
    baseline_precision = (results_baseline['y_pred'] == 1).sum() / len(results_baseline['y_pred'])
    ax.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=1.5, label='Baseline (No Skill)')

    ax.set_xlabel('Recall (True Positive Rate)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison: Random Forest Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    pr_comparison_path = output_dir / 'precision_recall_curves_comparison.png'
    plt.savefig(pr_comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Precision-Recall comparison plot saved to: {pr_comparison_path}")
    plt.close()


def analyze_thresholds(results_baseline, results_balanced, y_test):
    """
    Analyze Precision and Recall at different classification thresholds (0.0 to 1.0 in 0.1 steps).
    Shows metrics separated by class (No Failure and Failure).

    Parameters:
        results_baseline: Results dictionary from baseline model
        results_balanced: Results dictionary from balanced model
        y_test: Test target vector
    """
    print(f"\n{'='*70}")
    print("THRESHOLD ANALYSIS: Precision & Recall at Different Thresholds")
    print(f"{'='*70}\n")

    thresholds_to_test = np.arange(0.0, 1.1, 0.1)

    for model_name, results in [('Baseline', results_baseline), ('Balanced Weights', results_balanced)]:
        print(f"\n{model_name} Model:")
        print("-" * 100)

        threshold_data = []

        for threshold in thresholds_to_test:
            # Apply custom threshold
            y_pred_custom = (results['y_pred_proba'] >= threshold).astype(int)

            # Precision and Recall per class
            # Class 0: No Failure
            precision_0 = precision_score(y_test, y_pred_custom, labels=[0], average=None, zero_division=0)[0]
            recall_0 = recall_score(y_test, y_pred_custom, labels=[0], average=None, zero_division=0)[0]

            # Class 1: Failure
            if (y_pred_custom == 1).sum() == 0:
                precision_1 = 0.0
            else:
                precision_1 = precision_score(y_test, y_pred_custom, labels=[1], average=None, zero_division=0)[0]
            
            recall_1 = recall_score(y_test, y_pred_custom, labels=[1], average=None, zero_division=0)[0]

            threshold_data.append({
                'Threshold': f"{threshold:.1f}",
                'Prec (No Fail)': f"{precision_0:.4f}",
                'Recall (No Fail)': f"{recall_0:.4f}",
                'Prec (Fail)': f"{precision_1:.4f}",
                'Recall (Fail)': f"{recall_1:.4f}"
            })

        # Create and display DataFrame
        df_thresholds = pd.DataFrame(threshold_data)
        print(df_thresholds.to_string(index=False))
        print()


def plot_confusion_matrices(results_baseline, results_balanced):
    """
    Plot confusion matrices for both models.

    Parameters:
        results_baseline: Results dictionary from baseline model
        results_balanced: Results dictionary from balanced model
    """
    output_dir = Path('data/Feature Importance Analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results in [(ax1, results_baseline), (ax2, results_balanced)]:
        cm = results['confusion_matrix']
        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, str(cm[i, j]),
                             ha="center", va="center",
                             color="black", fontsize=14, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted: No Failure', 'Predicted: Failure'])
        ax.set_yticklabels(['Actual: No Failure', 'Actual: Failure'])
        ax.set_title(f"{results['model_name']}", fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrices_random_forest.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix plot saved to: {cm_path}")
    plt.close()


def main():
    """Main execution: Train and compare Random Forest models."""

    # Load and prepare data
    print("Loading data...")
    df = load_data("data/raw/ai4i2020.csv")
    df = create_engineered_features(df)

    # Define features (using extended features: baseline + new)
    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
        "Machine power [W]"
    ]
    target_column = "Machine failure"

    # Prepare X and y
    X = df[feature_columns]
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution in test set:")
    print(f"  No Failure: {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
    print(f"  Failure:    {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

    # ── MODEL 1: BASELINE ──────────────────────────────────────────────────────
    results_baseline = train_and_evaluate_rf(
        X_train, X_test, y_train, y_test,
        "Baseline Random Forest",
        class_weight=None
    )

    # ── MODEL 2: BALANCED CLASS WEIGHTS ────────────────────────────────────────
    results_balanced = train_and_evaluate_rf(
        X_train, X_test, y_train, y_test,
        "Random Forest (class_weight='balanced')",
        class_weight='balanced'
    )

    # ── COMPARISON ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")

    print(f"ROC-AUC Improvement: {(results_balanced['roc_auc'] - results_baseline['roc_auc']) * 100:+.2f}%")

    # Threshold Analysis
    analyze_thresholds(results_baseline, results_balanced, y_test)

    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")

    plot_roc_curves(results_baseline, results_balanced)
    plot_precision_recall_curves(results_baseline, results_balanced)
    plot_confusion_matrices(results_baseline, results_balanced)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll outputs saved to: data/Feature Importance Analysis/")


if __name__ == "__main__":
    main()
