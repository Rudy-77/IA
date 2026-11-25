"""Entrena Eigenfaces con TODO el dataset usando numpy (mucho más rápido).

Uso:
    python entrenar_eigen_rapido.py
    python entrenar_eigen_rapido.py --max-por-persona 300 --componentes 80

Hace el mismo cálculo que cv.face.EigenFaceRecognizer.train() (PCA sobre las
fotos de 100x100), pero el eigen de la matriz de 10000x10000 lo resuelve numpy
con todos los núcleos, en vez de OpenCV con uno solo (que tarda horas con
~15 000 fotos). Guarda el XML en el formato de OpenCV, así que
reconocer_eigen.py lo lee igual con EigenFaceRecognizer.read().
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
parser.add_argument("--max-por-persona", type=int, default=0,
                    help="imágenes por persona (0 = todas)")
parser.add_argument("--componentes", type=int, default=80,
                    help="eigenfaces a conservar")
parser.add_argument("--modelo", default="laloEigenface.xml")
parser.add_argument("--etiquetas", default="etiquetas_eigen.txt")
args = parser.parse_args()

t0 = time.perf_counter()
faces = sorted(d for d in os.listdir(args.dataset)
               if os.path.isdir(os.path.join(args.dataset, d)))
print("Personas:", faces, flush=True)

labels = []
faces_data = []
for label, person in enumerate(faces):
    person_path = os.path.join(args.dataset, person)
    archivos = sorted(f for f in os.listdir(person_path) if f.lower().endswith(EXT))
    elegidos = repartir(archivos, args.max_por_persona)
    n = 0
    for img_name in elegidos:
        img = cv.imread(os.path.join(person_path, img_name), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Igualar histograma, igual que entrenar_eigen.py y reconocer_eigen.py
        faces_data.append(cv.equalizeHist(cv.resize(img, TAM)).ravel())
        labels.append(label)
        n += 1
    print(f"  {person} (label={label}): {n} de {len(archivos)} imágenes", flush=True)

X = np.array(faces_data, dtype=np.float64)          # N x 10000
mean = X.mean(axis=0, keepdims=True)
Xc = X - mean
k = args.componentes
print(f"Calculando PCA con {len(X)} imágenes...", flush=True)

if len(X) > X.shape[1]:
    # Más fotos que píxeles: eigen de la covarianza 10000x10000
    vals, vecs = np.linalg.eigh(Xc.T @ Xc)
    orden = np.argsort(vals)[::-1][:k]
    vals, vecs = vals[orden], vecs[:, orden]
else:
    # Menos fotos que píxeles: truco de la matriz chica N x N
    vals, v = np.linalg.eigh(Xc @ Xc.T)
    orden = np.argsort(vals)[::-1][:k]
    vals, v = vals[orden], v[:, orden]
    vecs = Xc.T @ v
    vecs /= np.linalg.norm(vecs, axis=0, keepdims=True)

proyecciones = Xc @ vecs                             # N x k
t_pca = time.perf_counter()
print(f"PCA: {t_pca - t0:.1f} s", flush=True)

# Escribir con el mismo formato que EigenFaceRecognizer.write()
fs = cv.FileStorage(args.modelo, cv.FILE_STORAGE_WRITE)
fs.startWriteStruct("opencv_eigenfaces", cv.FILE_NODE_MAP)
fs.write("threshold", float(np.finfo(np.float64).max))
fs.write("num_components", k)
fs.write("mean", mean)
fs.write("eigenvalues", vals.reshape(-1, 1))
fs.write("eigenvectors", vecs)
fs.startWriteStruct("projections", cv.FILE_NODE_SEQ)
for p in proyecciones:
    fs.write("", p.reshape(1, -1))
fs.endWriteStruct()
fs.write("labels", np.array(labels, dtype=np.int32).reshape(-1, 1))
fs.startWriteStruct("labelsInfo", cv.FILE_NODE_MAP)
fs.endWriteStruct()
fs.endWriteStruct()
fs.release()

with open(args.etiquetas, "w", encoding="utf-8") as f:
    f.write("\n".join(faces))

mb = os.path.getsize(args.modelo) / 1e6
print(f"Modelo escrito: {args.modelo} ({mb:.0f} MB) | nombres en {args.etiquetas}")
print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
