"""
data_preprocessing.py
Phase 2: Datenvorverarbeitung und Feature Engineering.

Funktionen:
    create_engineered_features() – Neue Features aus bestehenden kombinieren
    analyze_new_features()       – Korrelation neuer Features mit Zielvariable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt neue Features durch Kombination bestehender Features.

    Neue Features:
    - Temperature difference [K]: Process temperature - Air temperature
    - Machine power [W]: Rotational speed [rpm] * Torque [Nm] * (2π/60) für W

    Parameter:
        df: Der ursprüngliche DataFrame

    Rückgabe:
        df: DataFrame mit zusätzlichen Features
    """
    df = df.copy()

    # Temperaturdifferenz: Prozesstemperatur - Lufttemperatur
    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    # Maschinenleistung: Drehzahl * Drehmoment
    # Umwandlung: rpm zu rad/s (rpm * 2π/60) und Nm zu W
    df["Machine power [W]"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)
    )

    print("=" * 55)
    print("NEUE FEATURES ERSTELLT")
    print("=" * 55)
    print(f"  Temperature difference [K]: Process temp - Air temp")
    print(f"  Machine power [W]: Rotational speed * Torque * (2π/60)")
    print(f"  → {len(df)} Zeilen, {len(df.columns)} Spalten")

    # Statistiken der neuen Features
    print("\nStatistiken der neuen Features:")
    for col in ["Temperature difference [K]", "Machine power [W]"]:
        if col in df.columns:
            print(f"  {col}:")
            print(f"    Mittelwert: {df[col].mean():.2f}")
            print(f"    Std-Abw:    {df[col].std():.2f}")
            print(f"    Min/Max:    {df[col].min():.2f} / {df[col].max():.2f}")

    return df


def analyze_new_features(df: pd.DataFrame, target_col: str = "Machine failure") -> None:
    """
    Analysiert die Korrelation der neuen Features mit der Zielvariable.

    Parameter:
        df: DataFrame mit neuen Features
        target_col: Name der Zielspalte
    """
    new_features = ["Temperature difference [K]", "Machine power [W]"]

    print("\n" + "=" * 55)
    print("KORRELATION NEUER FEATURES MIT ZIELVARIABLE")
    print("=" * 55)

    # Korrelation mit Zielvariable
    correlations = {}
    for feature in new_features:
        if feature in df.columns and target_col in df.columns:
            corr = df[feature].corr(df[target_col])
            correlations[feature] = corr
            print(f"  {feature:<30}: {corr:.4f}")

    # Vergleich mit bestehenden Features
    print("\nVergleich mit bestehenden Features:")
    existing_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]"
    ]

    for feature in existing_features:
        if feature in df.columns:
            corr = df[feature].corr(df[target_col])
            print(f"  {feature:<30}: {corr:.4f}")

    print("\nInterpretation:")
    for feature, corr in correlations.items():
        strength = "schwach" if abs(corr) < 0.1 else "mittel" if abs(corr) < 0.3 else "stark"
        direction = "positiv" if corr > 0 else "negativ"
        print(f"  {feature}: {strength} {direction}e Korrelation (r = {corr:.3f})")


def plot_new_features_correlation(df: pd.DataFrame, target_col: str = "Machine failure") -> None:
    """
    Erstellt eine Visualisierung der Korrelationen der neuen Features mit der Zielvariable.

    Parameter:
        df: DataFrame mit neuen Features
        target_col: Name der Zielspalte
    """
    new_features = ["Temperature difference [K]", "Machine power [W]"]
    existing_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]"
    ]

    all_features = new_features + existing_features
    correlations = {}

    for feature in all_features:
        if feature in df.columns:
            corr = df[feature].corr(df[target_col])
            correlations[feature] = corr

    # Sortiere nach absoluter Korrelation
    sorted_features = sorted(correlations.keys(), key=lambda x: abs(correlations[x]), reverse=True)
    sorted_corrs = [correlations[f] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(sorted_features)), sorted_corrs, color="#4C9BE8")

    # Markiere neue Features
    for i, feature in enumerate(sorted_features):
        if feature in new_features:
            bars[i].set_color("#E8694C")

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Korrelationskoeffizient")
    ax.set_title("Korrelation aller Features mit Machine failure")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Werte annotieren
    for i, value in enumerate(sorted_corrs):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f"{value:.3f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=10)

    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C9BE8", label="Bestehende Features"),
        Patch(facecolor="#E8694C", label="Neue Features")
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig("data/processed/new_features_correlation.png", dpi=150, bbox_inches="tight")
    print("  📊 Korrelationsdiagramm gespeichert: data/processed/new_features_correlation.png")
    plt.show()


# ── Direkt ausführbar ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Daten laden
    path = Path("data/raw/ai4i2020.csv")
    df = pd.read_csv(path)

    # Neue Features erstellen
    df_engineered = create_engineered_features(df)

    # Analyse der neuen Features
    analyze_new_features(df_engineered)

    # Visualisierung
    plot_new_features_correlation(df_engineered)
