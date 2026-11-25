"""Entrena Eigenfaces (versión del profe adaptada a este proyecto).

Uso:
    python entrenar_eigen.py
    python entrenar_eigen.py --max-por-persona 300 --componentes 80

Diferencias con el script del profe:
 - Usa la carpeta dataset/ de este proyecto (no /home/likcos/pruebacaras).
 - Eigenfaces exige que TODAS las imágenes midan lo mismo: se fuerza 100x100.
 - Eigenfaces hace PCA sobre todas las fotos juntas; con ~15 000 imágenes se
   tarda muchísimo y el XML pesa GB. Por eso se toman --max-por-persona fotos
   REPARTIDAS por persona y se limita el número de componentes.
 - Iguala el histograma de cada foto (Eigenfaces es muy sensible a la luz);
   reconocer_eigen.py hace lo mismo con la webcam.
 - Guarda etiquetas_eigen.txt para que reconocer_eigen.py use el mismo orden.
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
parser.add_argument("--max-por-persona", type=int, default=300,
                    help="imágenes por persona (0 = todas, puede tardar mucho)")
parser.add_argument("--componentes", type=int, default=80,
                    help="eigenfaces a conservar (0 = todas)")
parser.add_argument("--modelo", default="laloEigenface.xml")
parser.add_argument("--etiquetas", default="etiquetas_eigen.txt")
args = parser.parse_args()

t0 = time.perf_counter()
faces = sorted(d for d in os.listdir(args.dataset)
               if os.path.isdir(os.path.join(args.dataset, d)))
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
        # Igualar histograma: reduce la diferencia de luz entre video y webcam
        faces_data.append(cv.equalizeHist(cv.resize(img, TAM)))
        labels.append(label)
        n += 1
    print(f"  {person} (label={label}): {n} de {len(archivos)} imágenes")

print(f"Entrenando Eigenfaces con {len(faces_data)} imágenes...")
face_recognizer = cv.face.EigenFaceRecognizer_create(num_components=args.componentes)
face_recognizer.train(faces_data, np.array(labels))
face_recognizer.write(args.modelo)
with open(args.etiquetas, "w", encoding="utf-8") as f:
    f.write("\n".join(faces))

mb = os.path.getsize(args.modelo) / 1e6
print(f"Modelo escrito: {args.modelo} ({mb:.0f} MB) | nombres en {args.etiquetas}")
print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")
