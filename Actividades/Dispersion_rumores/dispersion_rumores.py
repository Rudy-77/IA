"""
Dispersión de rumores (#ExamenFiltrado) — misiones M1 a M6.

Uso:
    python dispersion_rumores.py                # busca los CSV en ./datasets
    python dispersion_rumores.py ruta/datasets  # o en otra carpeta

Espera estos archivos (los de la carpeta dispersion_rumores/datasets/ de la materia):
    eda_ventanas_10min.csv         M1
    similitud_tweets_tf.csv        M2
    timeline_tweets.csv            M3
    cascada_enlaces.csv            M3
    knn_cuentas_historial.csv      M4
    knn_cuentas_caso.csv           M4
    arbol_amplificacion_historial.csv   M5
    arbol_escenarios_caso.csv           M5
    regresion_alcance_rumor.csv    M6

No inventa ninguna fila: solo calcula sobre lo que haya en los CSV.
Imprime todas las tablas que pide cada misión y guarda la misma salida en
resultados_dispersion.txt para pegarla en el Org de entrega. Las respuestas
de interpretación (M0, M7 y las frases de cada misión) las redactas tú con
estos números.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets")
UMBRAL = 0.90
pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)


class Tee:
    """Escribe en pantalla y en un archivo a la vez."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


sys.stdout = Tee("resultados_dispersion.txt")


def titulo(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def leer(nombre):
    p = D / nombre
    if not p.exists():
        sys.exit(f"Falta {p}. Copia los CSV de la carpeta datasets/ de la materia.")
    return pd.read_csv(p)


def split_xy(train, new, target):
    """Features = columnas numéricas que están en train y en new (sin el target)."""
    comunes = [c for c in train.columns if c in new.columns and c != target]
    num = [c for c in comunes if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in comunes if c not in num and train[c].nunique() < len(train) * 0.5]
    X = pd.get_dummies(train[num + cat], columns=cat)
    Xn = pd.get_dummies(new[num + cat], columns=cat).reindex(columns=X.columns, fill_value=0)
    return X.astype(float), Xn.astype(float)


def col_id(df):
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]


# ---------------------------------------------------------------- M1
titulo("M1 — EDA por ventanas de 10 min")
v = leer("eda_ventanas_10min.csv").sort_values("ventana_min_ini")
print(v.to_string(index=False))
crit = v.loc[v["tweets"].idxmax()]
resto = v.drop(v["tweets"].idxmax())
print(f"\nVentana crítica: min {crit['ventana_min_ini']}–{crit['ventana_min_fin']}")
print(f"Tweets en esa ventana: {crit['tweets']} (mediana del resto: {resto['tweets'].median()})")
print(f"pct_cuentas_nuevas: {crit['pct_cuentas_nuevas']} (media del resto: {resto['pct_cuentas_nuevas'].mean():.2f})")

# ---------------------------------------------------------------- M2
titulo("M2 — Similitud coseno")
df = leer("similitud_tweets_tf.csv")
feats = [c for c in df.columns if c not in ("documento", "tipo")]
semillas = df[df["tipo"] == "semilla"].set_index("documento")[feats]
tweets = df[df["tipo"] == "tweet"].set_index("documento")[feats]
sim_sem = pd.DataFrame(cosine_similarity(tweets.values, semillas.values), index=tweets.index, columns=semillas.index)
sim_tw = pd.DataFrame(cosine_similarity(tweets.values), index=tweets.index, columns=tweets.index)
print("Coseno tweet × semilla:\n", sim_sem.round(3))
cerca = pd.DataFrame({"semilla_mas_cercana": sim_sem.idxmax(axis=1), "coseno": sim_sem.max(axis=1).round(3)})
print("\nSemilla más cercana por tweet:\n", cerca)
print(f"\nPares casi clon (coseno ≥ {UMBRAL}):")
ids = list(sim_tw.index)
pares = [(a, b, sim_tw.loc[a, b]) for i, a in enumerate(ids) for b in ids[i + 1:] if sim_tw.loc[a, b] >= UMBRAL]
for a, b, s in sorted(pares, key=lambda x: -x[2]):
    print(f"  {a} ~ {b}: {s:.3f}")
if not pares:
    print("  (ninguno con ese umbral; prueba otro y justifícalo en una línea)")

# ---------------------------------------------------------------- M3
titulo("M3 — Timeline y cascada")
t = leer("timeline_tweets.csv").sort_values("minuto")
print("Timeline ordenado por minuto (los primeros 15):\n", t.head(15).to_string(index=False))
print("\nCuentas más recientes (posibles bots):\n", t.sort_values("cuenta_dias").head(10).to_string(index=False))
print("\nCuentas más antiguas:\n", t.sort_values("cuenta_dias", ascending=False).head(5).to_string(index=False))
print("\nNOTA: filtra a mano solo los tweets del tema 'examen' según tu M2 antes de elegir al origen.")
c = leer("cascada_enlaces.csv")
print("\nEnlaces por tipo:\n", c.iloc[:, -1].value_counts() if c.shape[1] >= 3 else c)
print("\nCascada (cópiala al ASCII de tu entrega):")
orig, dest = c.columns[0], c.columns[1]
tipo = "tipo_enlace" if "tipo_enlace" in c.columns else c.columns[-1]
for _, r in c.iterrows():
    print(f"  @{r[orig]}  --{r[tipo]}-->  @{r[dest]}")

# ---------------------------------------------------------------- M4
titulo("M4 — k-NN: ¿bot o humano?")
train = leer("knn_cuentas_historial.csv")
caso = leer("knn_cuentas_caso.csv")
target = "etiqueta"
idc = "cuenta_id" if "cuenta_id" in train.columns else col_id(caso)
X, Xn = split_xy(train.drop(columns=[idc], errors="ignore"), caso.drop(columns=[idc], errors="ignore"), target)
y = train[target]
cv = int(min(5, y.value_counts().min()))
sc = StandardScaler().fit(X)
Xs, Xns = sc.transform(X), sc.transform(Xn)
res = {}
for k in (3, 5, 7):
    s = cross_val_score(KNeighborsClassifier(n_neighbors=k), Xs, y, cv=cv, scoring="f1_macro")
    res[k] = s.mean()
    print(f"k={k}: F1 macro CV = {s.mean():.3f} ± {s.std():.3f}  (cv={cv}, con escalado)")
mejor = max(res, key=res.get)
print(f"\nk elegido: {mejor}")
m = KNeighborsClassifier(n_neighbors=mejor).fit(Xs, y)
pred = pd.DataFrame({idc: caso[idc], "prediccion": m.predict(Xns)})
print(pred.to_string(index=False))
m0 = KNeighborsClassifier(n_neighbors=mejor).fit(X.values, y)
sin = m0.predict(Xn.values)
print(f"\nSin escalar, ¿cambia alguna predicción? {'SÍ' if (sin != pred['prediccion'].values).any() else 'no'}")

# ---------------------------------------------------------------- M5
titulo("M5 — Árbol de decisión (profundidad 4)")
h = leer("arbol_amplificacion_historial.csv")
e = leer("arbol_escenarios_caso.csv")
objetivos = [c for c in h.columns if c not in e.columns]
tgt = objetivos[-1] if objetivos else h.columns[-1]
print(f"Y detectada: {tgt}")
ide = col_id(e)
Xh, Xe = split_xy(h.drop(columns=[c for c in h.columns if c == ide and c not in e.columns], errors="ignore"), e, tgt)
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xh, h[tgt])
print(export_text(clf, feature_names=list(Xh.columns)))
print(pd.DataFrame({"escenario": e[ide], "accion_predicha": clf.predict(Xe)}).to_string(index=False))

# ---------------------------------------------------------------- M6
titulo("M6 — Regresión lineal del alcance")
r = leer("regresion_alcance_rumor.csv")
cols = ["minutos_desde_origen", "bots_activos", "desmentidos_activos"]
lr = LinearRegression().fit(r[cols], r["alcance_cuentas"])
print(f"β0 (intercepto) = {lr.intercept_:.3f}")
for n, b in zip(cols, lr.coef_):
    print(f"β {n} = {b:.3f}")
print(f"R² = {lr.score(r[cols], r['alcance_cuentas']):.3f}")
esc = pd.DataFrame([("A", 60, 10, 0), ("B", 60, 10, 2), ("C", 60, 0, 2)], columns=["escenario"] + cols)
esc["alcance_predicho"] = lr.predict(esc[cols]).round(1)
print("\n", esc.to_string(index=False))
print("\nFin. La salida quedó en resultados_dispersion.txt")
