"""
El Código Coilocito — Caso forense: Actos I, II, III y Acto Extra.

Uso:
    python coilocito.py                # busca los CSV en ./datasets
    python coilocito.py ruta/datasets  # o en otra carpeta

Espera estos archivos (carpeta codigo_coilocito/datasets/ de la materia):
    similitud_tf_documentos.csv        Acto I
    knn_historial_ataques.csv          Acto II
    knn_caso_arispe.csv                Acto II
    arbol_fugas_historial.csv          Acto III
    arbol_escenario_culpable.csv       Acto III
    arbol_escenario_valeriano.csv      Acto III (contrafactual)
    regresion_benchmark_coilocitos.csv Acto Extra

No inventa filas ni resultados. Imprime las tablas de cada acto y guarda la
misma salida en resultados_coilocito.txt. La hipótesis, la acusación, la orden
de interceptación y el veredicto los redactas tú a partir de estos números.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets")
PROFUNDIDAD = 4  # el enunciado pide entre 3 y 5
pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 200)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


sys.stdout = Tee("resultados_coilocito.txt")


def titulo(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def leer(nombre):
    p = D / nombre
    if not p.exists():
        sys.exit(f"Falta {p}. Copia los CSV de la carpeta datasets/ de la materia.")
    return pd.read_csv(p)


def objetivo(train, nuevo):
    """La Y es la columna que está en el historial y NO en el caso a predecir."""
    c = [x for x in train.columns if x not in nuevo.columns]
    return c[-1] if c else train.columns[-1]


def matriz(train, nuevos, target, ignorar=()):
    """Devuelve X de entrenamiento y las X de cada DataFrame nuevo, con las mismas columnas."""
    comunes = [c for c in train.columns if c != target and c not in ignorar and all(c in n.columns for n in nuevos)]
    num = [c for c in comunes if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in comunes if c not in num and train[c].nunique() < len(train) * 0.5]
    X = pd.get_dummies(train[num + cat], columns=cat).astype(float)
    out = [pd.get_dummies(n[num + cat], columns=cat).reindex(columns=X.columns, fill_value=0).astype(float) for n in nuevos]
    return X, out


def id_de(df):
    obj = [c for c in df.columns if df[c].dtype == object]
    return obj[0] if obj else df.columns[0]


# ---------------------------------------------------------------- Acto I
titulo("ACTO I — Similitud coseno: pistas vs perfiles")
df = leer("similitud_tf_documentos.csv")
feats = [c for c in df.columns if c not in ("documento", "tipo")]
perfiles = df[df["tipo"] == "perfil"].set_index("documento")[feats]
pistas = df[df["tipo"] == "pista"].set_index("documento")[feats]
sim = pd.DataFrame(cosine_similarity(pistas.values, perfiles.values), index=pistas.index, columns=perfiles.index)
print(sim.round(3))
print("\nMejor match por pista:")
print(pd.DataFrame({"perfil_mas_similar": sim.idxmax(axis=1), "coseno": sim.max(axis=1).round(3)}))
# Un par a mano, como pide el enunciado (primera pista con su mejor perfil)
a, b = pistas.index[0], sim.loc[pistas.index[0]].idxmax()
va, vb = pistas.loc[a].values.astype(float), perfiles.loc[b].values.astype(float)
num, den = float(va @ vb), float((va @ va) ** 0.5 * (vb @ vb) ** 0.5)
print(f"\nA mano ({a} vs {b}): a·b = {num:.3f}, ‖a‖‖b‖ = {den:.3f}, coseno = {num / den:.3f}")

# ---------------------------------------------------------------- Acto II
titulo("ACTO II — k-NN: tipo de ataque")
tr = leer("knn_historial_ataques.csv")
caso = leer("knn_caso_arispe.csv")
tgt = "etiqueta_ataque" if "etiqueta_ataque" in tr.columns else objetivo(tr, caso)
ids = [c for c in tr.columns if c.lower() in ("id", "ataque_id", "caso_id", "documento") or c.lower().endswith("_id")]
X, (Xn,) = matriz(tr, [caso], tgt, ignorar=ids)
y = tr[tgt]
cv = int(min(5, y.value_counts().min()))
sc = StandardScaler().fit(X)
Xs, Xns = sc.transform(X), sc.transform(Xn)
res = {}
for k in (3, 5, 7):
    s = cross_val_score(KNeighborsClassifier(n_neighbors=k), Xs, y, cv=cv, scoring="f1_macro")
    res[k] = s.mean()
    print(f"k={k}: F1 macro CV = {s.mean():.3f} ± {s.std():.3f} (cv={cv}, con escalado)")
mejor = max(res, key=res.get)
pred = KNeighborsClassifier(n_neighbors=mejor).fit(Xs, y).predict(Xns)
pred0 = KNeighborsClassifier(n_neighbors=mejor).fit(X.values, y).predict(Xn.values)
print(f"\nk elegido: {mejor}  |  F1 macro CV: {res[mejor]:.3f}")
print(f"Predicción caso Arispe (con escalado): {list(pred)}")
print(f"Predicción sin escalar:                {list(pred0)}  → ¿cambia? {'SÍ' if list(pred) != list(pred0) else 'no'}")

# ---------------------------------------------------------------- Acto III
titulo(f"ACTO III — Árbol de decisión (max_depth={PROFUNDIDAD})")
h = leer("arbol_fugas_historial.csv")
culp = leer("arbol_escenario_culpable.csv")
val = leer("arbol_escenario_valeriano.csv")
tg = objetivo(h, culp)
print(f"Y detectada: {tg}")
Xh, (Xc, Xv) = matriz(h, [culp, val], tg, ignorar=[c for c in h.columns if c.lower().endswith("_id")])
clf = DecisionTreeClassifier(max_depth=PROFUNDIDAD, random_state=42).fit(Xh, h[tg])
print(export_text(clf, feature_names=list(Xh.columns)))
print("Ruta predicha (culpable que acusas):", list(clf.predict(Xc)))
print("Contrafactual Valeriano → ruta:     ", list(clf.predict(Xv)))
print("\nRecuerda: el enunciado pide acusar con coherencia entre actos. Revisa que el")
print("escenario_culpable corresponda a la persona que acusas en tu veredicto.")

# ---------------------------------------------------------------- Extra
titulo("ACTO EXTRA — Regresión lineal: tiempo vs número de células")
r = leer("regresion_benchmark_coilocitos.csv")
filas = []
for metodo, g in r.groupby("metodo"):
    lr = LinearRegression().fit(g[["n_celulas"]], g["tiempo_inferencia_s"])
    t10k = float(lr.predict(pd.DataFrame({"n_celulas": [10000]}))[0])
    filas.append((metodo, round(float(lr.intercept_), 4), round(float(lr.coef_[0]), 6), round(t10k, 2),
                  int(g["n_celulas"].min()), int(g["n_celulas"].max())))
tab = pd.DataFrame(filas, columns=["metodo", "beta0", "beta1", "t@10k (s)", "n_min", "n_max"])
print(tab.to_string(index=False))
print("\nOJO: 10 000 células queda fuera del rango observado (n_min–n_max) si n_max es menor;")
print("eso es extrapolar, y el ajuste lineal puede no mantenerse. Dilo en la interpretación.")
print("\nFin. La salida quedó en resultados_coilocito.txt")
