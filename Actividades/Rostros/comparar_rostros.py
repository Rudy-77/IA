"""Compara Eigenfaces, Fisherfaces y LBPH con el MISMO split y las mismas fotos.

Uso:
    python comparar_rostros.py                       # los tres métodos
    python comparar_rostros.py --metodo lbph         # solo uno (lbph | eigen | fisher)
    python comparar_rostros.py --por-persona 500 --componentes-eigen 80

Qué hace (mismo protocolo que entrenar.py):
 1. Toma --por-persona fotos REPARTIDAS de cada persona (dataset/<persona>/*.jpg).
 2. Parte por BLOQUES: el primer 80 % de cada persona entrena y el 20 % final
    prueba. No es un reparto al azar, porque los frames seguidos de un video son
    casi idénticos y un reparto aleatorio inflaría la exactitud.
 3. Entrena cada método, mide el tiempo, evalúa foto por foto y guarda el modelo.
 4. Escribe resultados/<metodo>.json y la matriz de confusión.

Preprocesado (el mismo que ya usan tus scripts):
    Eigenfaces y Fisherfaces: 100x100 gris + equalizeHist (como entrenar_eigen.py)
    LBPH:                     100x100 gris sin ecualizar (como entrenar.py)

Requiere opencv-contrib-python (cv2.face).
"""
import argparse
import json
import os
import time

import cv2 as cv
import numpy as np

TAM = (100, 100)
EXT = (".jpg", ".jpeg", ".png")


def repartir(lista, n):
    """n elementos repartidos de forma pareja (o todos si hay menos)."""
    if n <= 0 or len(lista) <= n:
        return list(lista)
    idx = np.linspace(0, len(lista) - 1, n).round().astype(int)
    return [lista[i] for i in idx]


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="dataset")
ap.add_argument("--por-persona", type=int, default=500)
ap.add_argument("--prueba", type=float, default=0.2)
ap.add_argument("--metodo", default="todos", choices=["todos", "lbph", "eigen", "fisher"])
ap.add_argument("--componentes-eigen", type=int, default=80,
                help="0 = todas las componentes (modelo enorme y lento)")
ap.add_argument("--salida", default="resultados")
ap.add_argument("--modelos", default="modelos")
args = ap.parse_args()
os.makedirs(args.salida, exist_ok=True)
os.makedirs(args.modelos, exist_ok=True)

personas = sorted(d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d)))
imgs, labels, es_prueba = [], [], []
for lab, p in enumerate(personas):
    carpeta = os.path.join(args.dataset, p)
    fs = repartir(sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXT)), args.por_persona)
    ims = []
    for f in fs:
        im = cv.imread(os.path.join(carpeta, f), cv.IMREAD_GRAYSCALE)
        if im is not None:
            ims.append(cv.resize(im, TAM))
    corte = int(len(ims) * (1 - args.prueba))
    for i, im in enumerate(ims):
        imgs.append(im)
        labels.append(lab)
        es_prueba.append(i >= corte)
labels = np.array(labels)
es_prueba = np.array(es_prueba)
tr = [i for i in range(len(imgs)) if not es_prueba[i]]
te = [i for i in range(len(imgs)) if es_prueba[i]]
print(f"Personas: {personas}")
print(f"Fotos: {len(imgs)} | entrenamiento: {len(tr)} | prueba: {len(te)}", flush=True)


def correr(nombre, crear, preproc):
    X_tr = [preproc(imgs[i]) for i in tr]
    X_te = [preproc(imgs[i]) for i in te]
    rec = crear()
    t0 = time.perf_counter()
    rec.train(X_tr, labels[tr])
    t_train = time.perf_counter() - t0

    ruta = os.path.join(args.modelos, f"{nombre}.xml")
    rec.write(ruta)
    mb = os.path.getsize(ruta) / 1e6

    n = len(personas)
    matriz = np.zeros((n, n), dtype=int)
    conf_ok, conf_mal = [], []
    t0 = time.perf_counter()
    preds = [rec.predict(x) for x in X_te]
    t_pred = (time.perf_counter() - t0) / len(X_te) * 1000
    for (pred, conf), i in zip(preds, te):
        matriz[labels[i], pred] += 1
        (conf_ok if pred == labels[i] else conf_mal).append(conf)
    acc = float(np.trace(matriz) / matriz.sum())
    por_persona = {p: int(matriz[k, k]) for k, p in enumerate(personas)}
    pruebas_por_persona = {p: int(matriz[k].sum()) for k, p in enumerate(personas)}
    res = {
        "metodo": nombre, "exactitud": acc, "entrenamiento_s": t_train,
        "prediccion_ms": t_pred, "modelo_mb": mb, "aciertos_por_persona": por_persona,
        "pruebas_por_persona": pruebas_por_persona, "matriz": matriz.tolist(),
        "personas": personas, "n_train": len(tr), "n_test": len(te),
        "conf_aciertos_mediana": float(np.median(conf_ok)) if conf_ok else None,
        "conf_aciertos_p95": float(np.percentile(conf_ok, 95)) if conf_ok else None,
        "conf_errores_mediana": float(np.median(conf_mal)) if conf_mal else None,
    }
    with open(os.path.join(args.salida, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n=== {nombre} ===")
    print(f"Exactitud: {acc:.1%} | entrenamiento: {t_train:.1f} s | predicción: {t_pred:.2f} ms/foto | modelo: {mb:.1f} MB")
    print("Aciertos por persona:", {p: f"{por_persona[p]}/{pruebas_por_persona[p]}" for p in personas})
    print("Confianza mediana  aciertos:", res["conf_aciertos_mediana"], "| errores:", res["conf_errores_mediana"], flush=True)


igualar = lambda im: cv.equalizeHist(im)
crudo = lambda im: im
if args.metodo in ("todos", "lbph"):
    correr("lbph", lambda: cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8), crudo)
if args.metodo in ("todos", "eigen"):
    correr("eigen", lambda: cv.face.EigenFaceRecognizer_create(num_components=args.componentes_eigen), igualar)
if args.metodo in ("todos", "fisher"):
    correr("fisher", lambda: cv.face.FisherFaceRecognizer_create(), igualar)
