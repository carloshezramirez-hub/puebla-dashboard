import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from joblib import dump

DATA_DIR = Path("data")
HIST = DATA_DIR / "historicos.csv"
OUT  = DATA_DIR / "model.joblib"

df = pd.read_csv(HIST)
df.columns = [c.strip().upper() for c in df.columns]

needed = ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES","OBJETIVO"]
for c in needed:
    if c not in df.columns:
        raise SystemExit(f"Falta la columna {c} en historicos.csv")

X = df[["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"]].copy()
X = np.log1p(X)
y = df["OBJETIVO"].astype(int).values

# Robusto a datasets pequeños o desbalanceados
cls_counts = Counter(y)
min_count = min(cls_counts.values())
use_stratify = (min_count >= 2 and len(cls_counts) > 1 and len(y) >= 4)

if use_stratify:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
else:
    if len(y) >= 4:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        Xtr, ytr = X, y
        Xte, yte = None, None

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
).fit(Xtr, ytr)

if Xte is not None and yte is not None and len(yte) > 0:
    proba = clf.predict_proba(Xte)[:, 1]
    brier = brier_score_loss(yte, proba)
    print(f"Brier test: {brier:.4f} (más bajo es mejor)")
else:
    print("Dataset pequeño: se entrenó sin set de prueba/estratificación.")

dump({"model": clf, "features": list(X.columns)}, OUT)
print(f"Modelo guardado en {OUT}")
