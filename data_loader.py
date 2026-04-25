"""
data_loader.py
Phase 1: Datensatz laden, inspizieren und Zielverteilung visualisieren.

Funktionen:
    load_data()         – CSV laden und ersten Überblick ausgeben
    inspect_data()      – Datentypen, fehlende Werte, Statistiken
    summarize_target()  – Klassenverteilung analysieren und visualisieren
"""

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


@dataclass(frozen=True)
class FeatureInfo:
    description: str
    role: str  # 'input', 'output' oder 'identifier'


FEATURE_METADATA: Dict[str, FeatureInfo] = {
    "UDI": FeatureInfo(
        description="Eindeutiger Gerätebezeichner. Keine inhaltliche Bedeutung für das Modell.",
        role="identifier",
    ),
    "Product ID": FeatureInfo(
        description="Produktions- bzw. Maschinen-ID, beschreibt den Maschinentyp oder die Produktserie.",
        role="input",
    ),
    "Type": FeatureInfo(
        description="Produktqualitäts-Typ oder Kategorie (H, L, M).", 
        role="input",
    ),
    "Air temperature [K]": FeatureInfo(
        description="Lufttemperatur in Kelvin.",
        role="input",
    ),
    "Process temperature [K]": FeatureInfo(
        description="Prozesstemperatur in Kelvin.",
        role="input",
    ),
    "Rotational speed [rpm]": FeatureInfo(
        description="Drehzahl der Maschine in Umdrehungen pro Minute.",
        role="input",
    ),
    "Torque [Nm]": FeatureInfo(
        description="Drehmoment in Newtonmeter.",
        role="input",
    ),
    "Tool wear [min]": FeatureInfo(
        description="Verschleiß des Werkzeugs in Minuten.",
        role="input",
    ),
    "Machine failure": FeatureInfo(
        description="Zielvariable: 0 = kein Fehler, 1 = Maschinenausfall.",
        role="output",
    ),
    "TWF": FeatureInfo(
        description="Tool wear failure (Werkzeugverschleiß-Ausfall).", 
        role="output",
    ),
    "HDF": FeatureInfo(
        description="Heat dissipation failure (Wärmeabfuhr-Ausfall).", 
        role="output",
    ),
    "PWF": FeatureInfo(
        description="Power failure (Spannungs-/Stromausfall).", 
        role="output",
    ),
    "OSF": FeatureInfo(
        description="Overstrain failure (Überlast-Ausfall).", 
        role="output",
    ),
    "RNF": FeatureInfo(
        description="Random failure (zufälliger Ausfall).", 
        role="output",
    ),
}


def print_feature_metadata(metadata: Dict[str, FeatureInfo]) -> None:
    """Gibt die Beschreibung aller Features und ihre Rolle aus."""
    print("=" * 55)
    print("FEATURE METADATEN")
    print("=" * 55)
    for name, info in metadata.items():
        print(f"  {name:<25} [{info.role:<10}] {info.description}")
    print()


# ── 1. Daten laden ────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Lädt den Datensatz aus einer CSV-Datei und gibt einen ersten Überblick.

    Parameter:
        filepath: Pfad zur CSV-Datei (z.B. "data/raw/ai4i2020.csv")

    Rückgabe:
        df: Der geladene DataFrame
    """
    df = pd.read_csv(filepath)

    print("=" * 55)
    print("DATENSATZ GELADEN")
    print("=" * 55)
    print(f"  Zeilen:   {df.shape[0]:,}")   # Anzahl Datenpunkte
    print(f"  Spalten:  {df.shape[1]}")      # Anzahl Features
    print()
    print("Erste 5 Zeilen:")
    print(df.head())

    return df


# ── 2. Daten inspizieren ──────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """
    Gibt einen detaillierten Überblick über den DataFrame:
    - Datentypen jeder Spalte
    - Fehlende Werte (absolut und prozentual)
    - Statistische Kennzahlen (min, max, mean, std)

    Parameter:
        df: Der geladene DataFrame
    """
    print("=" * 55)
    print("DATENTYPEN PRO SPALTE")
    print("=" * 55)
    # dtypes zeigt uns ob Spalten numerisch, text (object) oder bool sind
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} {dtype}")

    print()
    print("=" * 55)
    print("FEHLENDE WERTE")
    print("=" * 55)
    missing = df.isnull().sum()          # Anzahl NaN pro Spalte
    missing_pct = (missing / len(df) * 100).round(2)

    if missing.sum() == 0:
        print("  ✅ Keine fehlenden Werte gefunden.")
    else:
        for col in missing[missing > 0].index:
            print(f"  {col:<35} {missing[col]:>5} fehlend  ({missing_pct[col]}%)")

    print()
    print("=" * 55)
    print("STATISTISCHE KENNZAHLEN (nur numerische Spalten)")
    print("=" * 55)
    # describe() gibt min, max, mean, std, Quartile für alle Zahlenspalten
    print(df.describe().round(2).to_string())

def plot_input_feature_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[int, int] = (14, 10),
    save_path: str = "data/processed/input_feature_distributions.png",
) -> None:
    """Zeigt eine Matrix mit der Verteilung der Input-Features."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input"
        ]

    columns = [col for col in columns if col in df.columns]
    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes_list, columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            ax.hist(df[col].dropna(), bins=30, color="#4C9BE8", edgecolor="white")
            ax.set_xlabel("Wert")
        else:
            counts = df[col].value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#4C9BE8", edgecolor="white")
            ax.set_xlabel("Kategorie")
            ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

        ax.set_title(col)
        ax.set_ylabel("Anzahl Samples")

    for ax in axes_list[n_features:]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Verteilung der Input-Features", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Input-Feature-Verteilung gespeichert: {save_path}")
    plt.show()


def plot_output_feature_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[int, int] = (12, 8),
    save_path: str = "data/processed/output_feature_distributions.png",
) -> None:
    """Zeigt eine Matrix mit Kreisdiagrammen für die Output-Features."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "output"
        ]

    columns = [col for col in columns if col in df.columns]
    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    colors = ["#4C9BE8", "#E8694C"]

    for ax, col in zip(axes_list, columns):
        counts = df[col].value_counts().sort_index()
        labels = [str(idx) for idx in counts.index]
        values = counts.values
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors[: len(values)],
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 9},
        )
        ax.set_title(col)

    for ax in axes_list[n_features:]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Verteilung der Output-Features", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Output-Feature-Verteilung gespeichert: {save_path}")
    plt.show()


def plot_input_boxplots(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] = (14, 8),
    save_path: str = "data/processed/input_boxplots.png",
) -> None:
    """Zeigt Boxplots für numerische Input-Features."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input" and pd.api.types.is_numeric_dtype(df[name])
        ]

    columns = [col for col in columns if col in df.columns]
    if not columns:
        print("  ⚠️  Keine numerischen Input-Features gefunden.")
        return

    fig, axes = plt.subplots(1, len(columns), figsize=figsize, sharey=False)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor="#4C9BE8", color="black"), medianprops=dict(color="red"))
        ax.set_title(col)
        ax.set_ylabel("Wert")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Boxplots der numerischen Input-Features", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Input-Boxplots gespeichert: {save_path}")
    plt.show()


def plot_input_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    save_path: str = "data/processed/input_correlation_heatmap.png",
) -> None:
    """Zeigt eine Korrelations-Heatmap für Input-Features.

    Kategorische Features wie "Product ID" und "Type" werden vor der
    Korrelationsberechnung als Zahlen kodiert.
    """
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input"
        ]

    columns = [col for col in columns if col in df.columns]
    if len(columns) < 2:
        print("  ⚠️  Nicht genügend Input-Features für Korrelationsanalyse.")
        return

    corr_df = df[columns].copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(corr_df[col]):
            corr_df[col] = pd.Categorical(corr_df[col]).codes

    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="left")
    ax.set_yticklabels(columns)

    # Annotate correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black", fontsize=10)

    plt.title("Korrelations-Heatmap der Input-Features (inkl. kodierter Kategorien)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Korrelations-Heatmap gespeichert: {save_path}")
    plt.show()


def plot_input_vs_target_boxplots(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    columns: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str = "data/processed/input_vs_target_boxplots.png",
) -> None:
    """Zeigt Boxplots der Input-Features gruppiert nach der Zielvariable."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input" and pd.api.types.is_numeric_dtype(df[name])
        ]

    columns = [col for col in columns if col in df.columns]
    if not columns:
        print("  Keine numerischen Input-Features gefunden.")
        return

    n_features = len(columns)
    figsize = figsize or (4 * n_features, 5)
    fig, axes = plt.subplots(1, n_features, figsize=figsize, sharey=False)
    if n_features == 1:
        axes = [axes]

    target_groups = df[target_col].unique()
    colors = ["#4C9BE8", "#E8694C"]

    for ax, col in zip(axes, columns):
        data_to_plot = [df[df[target_col] == target][col].dropna() for target in sorted(target_groups)]
        bp = ax.boxplot(data_to_plot, labels=sorted(target_groups), patch_artist=True)

        for patch, color in zip(bp["boxes"], colors[:len(target_groups)]):
            patch.set_facecolor(color)

        ax.set_title(col, fontsize=11)
        ax.set_xlabel(target_col)
        ax.set_ylabel("Wert")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Input-Features vs. {target_col}", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Input vs. Target-Boxplots gespeichert: {save_path}")
    plt.show()


def plot_input_vs_target_distributions(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    columns: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    save_path: str = "data/processed/input_vs_target_distributions.png",
) -> None:
    """Zeigt Histogramme mit KDE für Input-Features, gruppiert nach der Zielvariable."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input" and pd.api.types.is_numeric_dtype(df[name])
        ]

    columns = [col for col in columns if col in df.columns]
    if not columns:
        print("  Keine numerischen Input-Features gefunden.")
        return

    n_features = len(columns)
    figsize = figsize or (4 * n_features, 5)
    fig, axes = plt.subplots(1, n_features, figsize=figsize, sharey=False)
    if n_features == 1:
        axes = [axes]

    palette = {0: "#4C9BE8", 1: "#E8694C"}

    for ax, col in zip(axes, columns):
        sns.histplot(
            data=df,
            x=col,
            hue=target_col,
            kde=True,
            stat="density",
            ax=ax,
            palette=palette,
            bins=30,
        )
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("Dichte")
        ax.legend(title=target_col, labels=["Normal (0)", "Ausfall (1)"])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Verteilung der Input-Features vs. {target_col}", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Input vs. Target-Verteilungen gespeichert: {save_path}")
    plt.show()


def plot_input_target_correlations(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    columns: list[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: str = "data/processed/input_target_correlations.png",
) -> None:
    """Zeigt die Korrelation aller Input-Features mit der Zielvariable."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input"
        ]

    columns = [col for col in columns if col in df.columns]
    if target_col not in df.columns:
        print(f"  ⚠️  Zielspalte '{target_col}' nicht gefunden.")
        return

    if not columns:
        print("  ⚠️  Keine Input-Features für die Korrelationsanalyse gefunden.")
        return

    corr_df = df[columns + [target_col]].copy()
    for col in columns + [target_col]:
        if not pd.api.types.is_numeric_dtype(corr_df[col]):
            corr_df[col] = pd.Categorical(corr_df[col]).codes

    corr_with_target = corr_df.corr()[target_col].drop(target_col)
    corr_with_target = corr_with_target.reindex(
        corr_with_target.abs().sort_values(ascending=False).index
    )

    print("\nKorrelationswerte mit Machine failure:")
    for feature, value in corr_with_target.items():
        print(f"  {feature:<25}: {value:.4f}")

    plot_height = max(figsize[1], 0.8 * len(corr_with_target))
    fig, ax = plt.subplots(figsize=(figsize[0], plot_height))
    y_pos = list(range(len(corr_with_target)))
    ax.barh(y_pos, corr_with_target.values, color="#4C9BE8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_with_target.index)
    ax.invert_yaxis()
    ax.set_xlabel("Korrelationskoeffizient")
    ax.set_title(f"Korrelation aller Input-Features mit {target_col}")
    ax.grid(True, axis="x", alpha=0.3)

    for i, value in enumerate(corr_with_target.values):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                color="black",
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Input-Target-Korrelationsgrafik gespeichert: {save_path}")
    plt.show()


def plot_high_correlation_scatterplots(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    columns: list[str] | None = None,
    correlation_threshold: float = 0.7,
    figsize: tuple[int, int] | None = None,
    save_path: str = "data/processed/high_correlation_scatterplots.png",
) -> None:
    """Zeigt Scatterplots für Input-Feature-Paare mit hoher Korrelation, getrennt nach Zielvariable."""
    if columns is None:
        columns = [
            name
            for name, info in FEATURE_METADATA.items()
            if info.role == "input" and pd.api.types.is_numeric_dtype(df[name])
        ]

    columns = [col for col in columns if col in df.columns]
    
    # Berechne Korrelationsmatrix
    corr_matrix = df[columns].corr().abs()
    
    # Finde Paare mit hoher Korrelation (ohne Diagonale)
    high_corr_pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if corr_matrix.iloc[i, j] >= correlation_threshold:
                high_corr_pairs.append((columns[i], columns[j], corr_matrix.iloc[i, j]))
    
    if not high_corr_pairs:
        print(f"  Keine Feature-Paare mit Korrelation >= {correlation_threshold} gefunden.")
        return

    # Sortiere nach Korrelation (absteigend)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    n_pairs = len(high_corr_pairs)
    figsize = figsize or (10, 5 * n_pairs)
    
    # Layout: n_pairs Zeilen, 2 Spalten (für Normal und Ausfall)
    fig, axes = plt.subplots(n_pairs, 2, figsize=figsize)
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    target_values = sorted(df[target_col].unique())
    labels = {0: "Normal (0)", 1: "Ausfall (1)"}
    colors = ["#4C9BE8", "#E8694C"]
    
    for row_idx, (x_col, y_col, corr_val) in enumerate(high_corr_pairs):
        for col_idx, target_val in enumerate(target_values):
            ax = axes[row_idx, col_idx]
            subset = df[df[target_col] == target_val]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                color=colors[col_idx],
                alpha=0.6,
                s=50,
                label=labels[target_val],
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col} ({labels[target_val]})\n(Korrelation: {corr_val:.3f})", fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"Scatterplots hochkorrelierter Input-Features (r >= {correlation_threshold})", fontsize=14, y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Hochkorrelierte Scatterplots gespeichert: {save_path}")
    plt.show()

# ── 3. Zielverteilung analysieren ─────────────────────────────────────────────

def summarize_target(df: pd.DataFrame, target_col: str = "Machine failure") -> None:
    """
    Analysiert die Verteilung der Zielvariable (normal vs. Ausfall).
    Gibt absolute Zahlen, Prozentwerte aus und erstellt eine Visualisierung.

    Parameter:
        df:         Der geladene DataFrame
        target_col: Name der Zielspalte (Standard: "Machine failure")
    """
    print("=" * 55)
    print(f"ZIELVERTEILUNG: '{target_col}'")
    print("=" * 55)

    counts = df[target_col].value_counts().sort_index()
    labels = {0: "Normal (0)", 1: "Ausfall (1)"}

    for val, count in counts.items():
        pct = count / len(df) * 100
        label = labels.get(val, str(val))
        print(f"  {label:<20} {count:>6,} Einträge  ({pct:.1f}%)")

    print()

    # Klassisches Problem im ML: Wenn eine Klasse viel seltener ist
    # (hier: Ausfälle), spricht man von einem "unbalancierten Datensatz".
    # Das beeinflusst später die Modellwahl und die Metriken.
    ratio = counts[0] / counts[1]
    print(f"  ⚠️  Klassen-Verhältnis Normal:Ausfall = {ratio:.0f}:1")
    print("  → Datensatz ist unbalanciert – relevant für Modellwahl!")

    # ── Visualisierung ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Verteilung der Zielvariable: Machine Failure", fontsize=13)

    bar_labels = [labels[v] for v in counts.index]
    colors = ["#4C9BE8", "#E8694C"]

    # Balkendiagramm – absolute Häufigkeiten
    axes[0].bar(bar_labels, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Absolute Häufigkeit")
    axes[0].set_ylabel("Anzahl Datenpunkte")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=10)

    # Tortendiagramm – prozentuale Verteilung
    axes[1].pie(
        counts.values,
        labels=bar_labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Prozentuale Verteilung")

    plt.tight_layout()
    plt.savefig("data/processed/target_distribution.png", dpi=150, bbox_inches="tight")
    print("  📊 Grafik gespeichert: data/processed/target_distribution.png")
    plt.show()


# ── Direkt ausführbar ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/raw/ai4i2020.csv")
    print()
    inspect_data(df)
    print()
    print_feature_metadata(FEATURE_METADATA)
    plot_input_feature_distributions(df)
    plot_output_feature_distributions(df)
    plot_input_boxplots(df)
    plot_input_correlation_heatmap(df)
    plot_input_vs_target_boxplots(df)
    plot_input_vs_target_distributions(df)
    plot_input_target_correlations(df)
    plot_high_correlation_scatterplots(df)
    print()
    summarize_target(df)
