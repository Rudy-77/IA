"""Entrena LBPH con el dataset y lo evalúa.

Uso:
    python entrenar.py
    python entrenar.py --max-por-persona 800 --muestra-prueba 100 --grid 8

Pasos:
 1. Carga dataset/<persona>/*.jpg (gris, 100x100). Si una persona tiene más de
    --max-por-persona imágenes, se toman REPARTIDAS a lo largo de todas las suyas
    (no las primeras), para conservar la variedad y balancear las clases.
 2. Evaluación: entrena con el primer 80% de cada persona y prueba con el 20%
    final. Se parte por BLOQUES (no al azar) porque los frames consecutivos de
    un video son casi idénticos; un reparto aleatorio inflaría la precisión.
    La prueba usa como máximo --muestra-prueba imágenes por persona.
 3. Reentrena con TODAS las imágenes elegidas y guarda modelo_lbph.xml +
    etiquetas.txt.
 4. Comandos para probar el modelo:
    python reconocer.py
 5. Numero de entreno 2

Por qué importa el tamaño: LBPH guarda el histograma de CADA imagen de
entrenamiento (grid*grid*256 números por imagen). OpenCV no puede leer XML de
más o menos 1.5 GB, así que hay que limitar las imágenes o la rejilla.
"""
import argparse
import os
import time

import cv2 as cv
import numpy as np

TAM = (100, 100)
EXT = (".jpg", ".jpeg", ".png")


def repartir(lista, n):
    """Devuelve n elementos repartidos de forma pareja (o todos si hay menos)."""
    if n <= 0 or len(lista) <= n:
        return list(lista)
    idx = np.linspace(0, len(lista) - 1, n).round().astype(int)
    return [lista[i] for i in idx]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--prueba", type=float, default=0.2,
                    help="fracción final de cada persona usada para prueba")
parser.add_argument("--max-por-persona", type=int, default=800,
                    help="imágenes por persona (0 = todas)")
parser.add_argument("--muestra-prueba", type=int, default=100,
                    help="imágenes de prueba por persona al evaluar (0 = todas)")
parser.add_argument("--grid", type=int, default=8,
                    help="celdas por lado de la rejilla LBPH (8 = 64 celdas, "
                         "4 = 16 celdas y un modelo 4 veces más chico)")
parser.add_argument("--modelo", default="modelo_lbph.xml")
parser.add_argument("--etiquetas", default="etiquetas.txt")
args = parser.parse_args()

t0 = time.perf_counter()

personas = sorted(d for d in os.listdir(args.dataset)
                  if os.path.isdir(os.path.join(args.dataset, d)))
if len(personas) < 2:
    raise SystemExit("Se necesitan al menos 2 personas en el dataset.")
print("Personas:", personas)

X_train, y_train = [], []
X_test_all, y_test_all = [], []   # todas las de prueba (van al modelo final)
X_eval, y_eval = [], []           # muestra de prueba usada solo para evaluar
for label, persona in enumerate(personas):
    carpeta = os.path.join(args.dataset, persona)
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT))
    total = len(archivos)
    archivos = repartir(archivos, args.max_por_persona)
    imgs = []
    for f in archivos:
        img = cv.imread(os.path.join(carpeta, f), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imgs.append(cv.resize(img, TAM))
    corte = int(len(imgs) * (1 - args.prueba))
    train, test = imgs[:corte], imgs[corte:]
    X_train += train
    y_train += [label] * len(train)
    X_test_all += test
    y_test_all += [label] * len(test)
    muestra = repartir(test, args.muestra_prueba)
    X_eval += muestra
    y_eval += [label] * len(muestra)
    print(f"  {persona} (label={label}): {len(imgs)} de {total} imágenes "
          f"({len(train)} train / {len(test)} test, evalúo {len(muestra)})")
t_carga = time.perf_counter()
print(f"Carga: {t_carga - t0:.1f} s")

# ---------- Evaluación ----------
g = args.grid
rec = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=g, grid_y=g)
rec.train(X_train, np.array(y_train))
t_train = time.perf_counter()
print(f"Entrenamiento de evaluación: {t_train - t_carga:.1f} s")

n = len(personas)
matriz = np.zeros((n, n), dtype=int)
distancias = []
for img, real in zip(X_eval, y_eval):
    pred, dist = rec.predict(img)
    matriz[real, pred] += 1
    distancias.append(dist)
t_eval = time.perf_counter()
print(f"Evaluación ({len(X_eval)} imágenes): {t_eval - t_train:.1f} s")

acc = np.trace(matriz) / matriz.sum()
print(f"\nPrecisión en prueba: {acc:.2%}")
print(f"Distancia media (confidence): {np.mean(distancias):.1f}  "
      f"| máx: {np.max(distancias):.1f}")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print("            " + " ".join(f"{p[:8]:>8}" for p in personas))
for i, p in enumerate(personas):
    print(f"{p[:10]:>10}  " + " ".join(f"{v:8d}" for v in matriz[i]))

# ---------- Modelo final con todas las imágenes elegidas ----------
rec_final = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=g, grid_y=g)
rec_final.train(X_train + X_test_all, np.array(y_train + y_test_all))
rec_final.write(args.modelo)
with open(args.etiquetas, "w", encoding="utf-8") as f:
    f.write("\n".join(personas))
t_fin = time.perf_counter()
mb = os.path.getsize(args.modelo) / 1e6
print(f"\nModelo guardado en {args.modelo} ({mb:.0f} MB) y nombres en {args.etiquetas}")
if mb > 1000:
    print("AVISO: el modelo pasa de ~1 GB y OpenCV podría no poder abrirlo.\n"
          "       Reentrena con menos fotos, por ejemplo: "
          "python entrenar.py --max-por-persona 600")
print(f"Tiempo total: {t_fin - t0:.1f} s")
