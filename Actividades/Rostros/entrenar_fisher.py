"""Entrena Fisherfaces (PCA + LDA) con el dataset de este proyecto.

Uso:
    python entrenar_fisher.py
    python entrenar_fisher.py --max-por-persona 400

Es el listado 3 de la materia, adaptado igual que entrenar_eigen.py:
 - Usa la carpeta dataset/ de este proyecto.
 - Fuerza 100x100 (todas las fotos deben medir lo mismo) e iguala el
   histograma, para que reconocer_fisher.py preprocese igual la webcam.
 - Toma --max-por-persona fotos REPARTIDAS por persona. Fisherfaces hace PCA y
   luego un eigen de una matriz de (N - personas) x (N - personas); con miles de
   fotos tarda mucho (con 2 800 fotos, o sea 400 por persona, fueron unos 25 min
   en las pruebas), así que no conviene usar las 15 000. Con --max-por-persona 200
   debería bajar bastante (no lo medí).
 - Guarda etiquetas_fisher.txt para que reconocer_fisher.py use el mismo orden.

El modelo sale muy chico (~2 MB): cada cara queda reducida a (personas - 1)
números, porque LDA no puede dar más ejes que clases menos una.
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
parser.add_argument("--max-por-persona", type=int, default=400,
                    help="imágenes por persona (0 = todas, puede tardar muchísimo)")
parser.add_argument("--modelo", default="laloFisherFace.xml")
parser.add_argument("--etiquetas", default="etiquetas_fisher.txt")
args = parser.parse_args()

t0 = time.perf_counter()
faces = sorted(d for d in os.listdir(args.dataset)
               if os.path.isdir(os.path.join(args.dataset, d)))
if len(faces) < 2:
    raise SystemExit("Fisherfaces necesita al menos 2 personas en el dataset.")
print("Personas:", faces)

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
        # Igualar histograma, igual que entrenar_eigen.py y reconocer_fisher.py
        faces_data.append(cv.equalizeHist(cv.resize(img, TAM)))
        labels.append(label)
        n += 1
    print(f"  {person} (label={label}): {n} de {len(archivos)} imágenes")

print(f"Entrenando Fisherfaces con {len(faces_data)} imágenes (puede tardar varios minutos)...",
      flush=True)
face_recognizer = cv.face.FisherFaceRecognizer_create()
face_recognizer.train(faces_data, np.array(labels))
face_recognizer.write(args.modelo)
with open(args.etiquetas, "w", encoding="utf-8") as f:
    f.write("\n".join(faces))

mb = os.path.getsize(args.modelo) / 1e6
print(f"Modelo escrito: {args.modelo} ({mb:.1f} MB) | nombres en {args.etiquetas}")
print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
