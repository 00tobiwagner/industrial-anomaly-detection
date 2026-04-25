# industrial-anomaly-detection
ML project: Quality control and anomaly detection

## 📦 Boxplot Analysis – Numerical Features

### 🔍 What is a Boxplot?

A boxplot (box-and-whisker plot) is a compact way to visualize the distribution of numerical data. It highlights central tendency, variability, and potential outliers.

**Key components:**

- **Median (Q2)**: The line inside the box; represents the 50th percentile.
- **Box (IQR – Interquartile Range)**:  
  The range between the 25th percentile (Q1) and 75th percentile (Q3).  
  → Contains the middle 50% of the data.
- **Whiskers**:  
  Extend to the smallest and largest values within the range:  
  `Q1 - 1.5 × IQR` and `Q3 + 1.5 × IQR`
- **Outliers**:  
  Data points outside the whisker range, plotted as individual points.

---

### 📊 Purpose of the Boxplot in this Analysis

The boxplots are used to:

- Understand the distribution of each numerical feature
- Identify variability and spread
- Detect potential outliers
- Compare different features on a statistical level

---

### 📈 Observations from the Boxplots

#### 1. Air Temperature [K]
- Narrow interquartile range → low variability
- Median centered within the box
- Few to no visible outliers

#### 2. Process Temperature [K]
- Similar distribution to air temperature
- Slightly higher median
- Low spread and minimal outliers

#### 3. Rotational Speed [rpm]
- Wider spread compared to temperature features
- Significant number of high-value outliers
- Distribution appears right-skewed

#### 4. Torque [Nm]
- Moderate spread with noticeable variability
- Presence of both low and high outliers
- Median slightly below center → potential skewness

#### 5. Tool Wear [min]
- Large spread across values
- Wide interquartile range
- No extreme outlier clustering, but high variability

---

### 🧠 Key Takeaways (Descriptive Only)

- Temperature features show low variance and stable distributions
- Rotational speed and torque exhibit higher variability and multiple outliers
- Tool wear spans a broad range of values, indicating strong dispersion

> Note: This section focuses on descriptive statistical analysis only.  
> No assumptions or predictive interpretations are made at this stage.

---

## 📌 Klassifikationskennzahlen (Accuracy, Precision, Recall, F1-Score)

Für das Modell werden folgende Metriken verwendet:

- **Accuracy**: Anteil der korrekt vorhergesagten Fälle an allen Fällen.
  - Beispiel: Wenn 98 von 100 Vorhersagen korrekt sind, ist die Accuracy 98%.
- **Precision**: Anteil der als positiv vorhergesagten Fälle, die tatsächlich positiv sind.
  - Wichtig bei Fehlervermeidung, wenn falsche Alarme vermieden werden sollen.
- **Recall**: Anteil der tatsächlichen positiven Fälle, die vom Modell erkannt werden.
  - Wichtig, wenn keine positiven Fälle übersehen werden dürfen.
- **F1-Score**: Harmonisches Mittel von Precision und Recall.
  - Gibt eine ausgewogene Gesamtbewertung, wenn sowohl Genauigkeit als auch Vollständigkeit wichtig sind.

Diese Kennzahlen helfen zu verstehen, wie gut das Modell zwischen normalem Betrieb und Ausfall unterscheidet, ohne nur auf die reine Trefferquote zu schauen.

---

## 📈 ROC Curve (Receiver Operating Characteristic)

Die **ROC Curve** visualisiert die Kompromisse zwischen **True Positive Rate (TPR)** und **False Positive Rate (FPR)** bei verschiedenen Klassifikationsschwellwerten.

### 🔍 Was ist die ROC Curve?

- **X-Achse (FPR)**: False Positive Rate = Falsche Alarme / alle tatsächlichen Negativen
  - Wie viele Normal-Fälle wurden fälschlicherweise als Ausfall klassifiziert?
- **Y-Achse (TPR)**: True Positive Rate = Korrekt erkannte Ausfälle / alle tatsächlichen Ausfälle
  - Wie viele echte Ausfälle hat das Modell erkannt?

### 📊 Interpretation der ROC Curve

- **Perfektes Modell**: Kurve geht nach oben (TPR=1) und dann nach rechts (FPR=0)
- **Zufälliges Modell**: Diagonale Linie von unten-links zu oben-rechts (keine Diskriminationsfähigkeit)
- **Reales Modell**: Kurve liegt zwischen diesen Extremen

### 📌 ROC-AUC Score

- **AUC (Area Under the Curve)**: Flächeninhalt unter der ROC Kurve
- Bereich: 0 bis 1
  - **0.5**: Zufälliges Raten (schlecht)
  - **0.7–0.8**: Fair
  - **0.8–0.9**: Gut
  - **0.9–1.0**: Sehr gut
- Je höher die AUC, desto besser wird die Klasse zwischen Ausfall und Nicht-Ausfall unterschieden.

### 💡 Warum ROC statt nur Accuracy?

- **Accuracy** kann irreführend sein, besonders bei **unausgewogenen Klassen** (z.B. 96% normal, 4% Ausfall)
- **ROC Curve** zeigt, wie viele Ausfälle erkannt werden, ohne zu viele falschen Alarme zu erzeugen
- Für Predictive Maintenance ist dies critical: Wir brauchen Ausfall-Erkennung ohne Übersensibilität