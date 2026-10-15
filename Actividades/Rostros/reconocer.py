"""Reconocimiento en vivo con webcam usando el modelo LBPH entrenado.

Uso:
    python reconocer.py
    python reconocer.py --umbral 80 --camara 0

confidence es una DISTANCIA: menor = más parecido. Si es >= umbral se muestra
"Desconocido". Calibra el umbral mirando los valores en pantalla con las
personas conocidas y con alguien que no esté en el dataset.
"""
import argparse
import os

import cv2 as cv

TAM = (100, 100)

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="modelo_lbph.xml")
parser.add_argument("--etiquetas", default="etiquetas.txt")
parser.add_argument("--umbral", type=float, default=80)
parser.add_argument("--camara", type=int, default=0)
args = parser.parse_args()

with open(args.etiquetas, encoding="utf-8") as f:
    personas = [l.strip() for l in f if l.strip()]

rec = cv.face.LBPHFaceRecognizer_create()
rec.read(args.modelo)

NOMBRE_XML = "haarcascade_frontalface_alt2.xml"
ruta_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOMBRE_XML)
if not os.path.exists(ruta_xml):  # si no está junto al script, usar el de OpenCV
    ruta_xml = cv.data.haarcascades + NOMBRE_XML
cascade = cv.CascadeClassifier(ruta_xml)
if cascade.empty():
    raise SystemExit(
        f"No se encontró {NOMBRE_XML}. Descárgalo en la carpeta del script:\n"
        f"curl -L -O https://raw.githubusercontent.com/opencv/opencv/4.x/"
        f"data/haarcascades/{NOMBRE_XML}"
    )

cap = cv.VideoCapture(args.camara)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    for (x, y, w, h) in cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60)):
        roi = cv.resize(gray[y:y + h, x:x + w], TAM, interpolation=cv.INTER_CUBIC)
        label_id, confidence = rec.predict(roi)

        if confidence < args.umbral:
            texto = f"{personas[label_id]} ({confidence:.0f})"
            color = (0, 255, 0)
        else:
            texto = f"Desconocido ({confidence:.0f})"
            color = (0, 0, 255)

        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, texto, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2, cv.LINE_AA)

    cv.imshow("LBPH (ESC para salir)", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
