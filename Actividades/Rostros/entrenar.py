"""Entrena LBPH con el dataset y lo evalúa.

Uso:
    python entrenar.py
    python entrenar.py --dataset dataset --prueba 0.2

Pasos:
 1. Carga dataset/<persona>/*.jpg (gris, 100x100).
 2. Evaluación: entrena con el primer 80% de cada persona y prueba con el 20%
    final. Se parte por BLOQUES (no al azar) porque los frames consecutivos de
    un video son casi idénticos; un reparto aleatorio inflaría la precisión.
 3. Reentrena con TODAS las imágenes y guarda modelo_lbph.xml + etiquetas.txt.
 4. Comandos para probar el modelo:
    python reconocer.py
 5. Numero de entreno 2 
"""
import argparse
import os

import cv2 as cv
import numpy as np

TAM = (100, 100)
EXT = (".jpg", ".jpeg", ".png")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--prueba", type=float, default=0.2,
                    help="fracción final de cada persona usada para prueba")
parser.add_argument("--modelo", default="modelo_lbph.xml")
parser.add_argument("--etiquetas", default="etiquetas.txt")
args = parser.parse_args()

personas = sorted(d for d in os.listdir(args.dataset)
                  if os.path.isdir(os.path.join(args.dataset, d)))
if len(personas) < 2:
    raise SystemExit("Se necesitan al menos 2 personas en el dataset.")
print("Personas:", personas)

X_train, y_train, X_test, y_test = [], [], [], []
for label, persona in enumerate(personas):
    carpeta = os.path.join(args.dataset, persona)
    archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT))
    imgs = []
    for f in archivos:
        img = cv.imread(os.path.join(carpeta, f), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imgs.append(cv.resize(img, TAM))
    corte = int(len(imgs) * (1 - args.prueba))
    X_train += imgs[:corte]
    y_train += [label] * corte
    X_test += imgs[corte:]
    y_test += [label] * (len(imgs) - corte)
    print(f"  {persona} (label={label}): {len(imgs)} imágenes "
          f"({corte} train / {len(imgs) - corte} test)")

# ---------- Evaluación ----------
rec = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
rec.train(X_train, np.array(y_train))

n = len(personas)
matriz = np.zeros((n, n), dtype=int)
distancias = []
for img, real in zip(X_test, y_test):
    pred, dist = rec.predict(img)
    matriz[real, pred] += 1
    distancias.append(dist)

acc = np.trace(matriz) / matriz.sum()
print(f"\nPrecisión en prueba: {acc:.2%}")
print(f"Distancia media (confidence): {np.mean(distancias):.1f}  "
      f"| máx: {np.max(distancias):.1f}")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print("            " + " ".join(f"{p[:8]:>8}" for p in personas))
for i, p in enumerate(personas):
    print(f"{p[:10]:>10}  " + " ".join(f"{v:8d}" for v in matriz[i]))

# ---------- Modelo final con todos los datos ----------
rec_final = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
rec_final.train(X_train + X_test, np.array(y_train + y_test))
rec_final.write(args.modelo)
with open(args.etiquetas, "w", encoding="utf-8") as f:
    f.write("\n".join(personas))
print(f"\nModelo guardado en {args.modelo} y nombres en {args.etiquetas}")
