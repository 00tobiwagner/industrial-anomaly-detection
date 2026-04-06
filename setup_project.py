"""
setup_project.py
Einmalig ausführen um:
1. Ordnerstruktur anzulegen
2. Datensatz herunterzuladen
"""

import os
import urllib.request

# ── 1. Ordnerstruktur ────────────────────────────────────────────────────────

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "tests",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # .gitkeep damit leere Ordner von git getrackt werden
    gitkeep = os.path.join(folder, ".gitkeep")
    if not os.listdir(folder):
        open(gitkeep, "w").close()

print("✅ Ordnerstruktur angelegt")

# ── 2. Datensatz herunterladen ───────────────────────────────────────────────

url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00601/ai4i2020.csv"
)
destination = "data/raw/ai4i2020.csv"

if os.path.exists(destination):
    print(f"ℹ️  Datensatz existiert bereits: {destination}")
else:
    print("⏳ Lade Datensatz herunter...")
    urllib.request.urlretrieve(url, destination)
    print(f"✅ Datensatz gespeichert: {destination}")

print("\n🚀 Setup abgeschlossen")