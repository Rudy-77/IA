"""Reconocimiento en vivo: Webcam + Haar + modelo Eigenfaces (script del profe).

Uso:
    python reconocer_eigen.py
    python reconocer_eigen.py --umbral 4500 --camara 0

confidence es una DISTANCIA: menor = más parecido. En Eigenfaces los valores
son de miles (no de decenas como en LBPH). Mira los números en pantalla con
personas conocidas y desconocidas y ajusta --umbral (bajar = más estricto).
"""
import argparse
import os

import cv2 as cv

TAM = (100, 100)

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="laloEigenface.xml")
parser.add_argument("--etiquetas", default="etiquetas_eigen.txt")
parser.add_argument("--umbral", type=float, default=4500)
parser.add_argument("--camara", type=int, default=0)
args = parser.parse_args()

with open(args.etiquetas, encoding="utf-8") as f:
    faces = [l.strip() for l in f if l.strip()]

face_recognizer = cv.face.EigenFaceRecognizer_create()
face_recognizer.read(args.modelo)

NOMBRE_XML = "haarcascade_frontalface_alt2.xml"
ruta_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOMBRE_XML)
if not os.path.exists(ruta_xml):  # si no está junto al script, usar el de OpenCV
    ruta_xml = cv.data.haarcascades + NOMBRE_XML
cascade = cv.CascadeClassifier(ruta_xml)

cap = cv.VideoCapture(args.camara)
if not cap.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

print(f"Modelo cargado: {', '.join(faces)}. Umbral: {args.umbral:g}. ESC para salir.")

fallos = 0
while True:
    ret, frame = cap.read()
    if not ret:
        # La cámara de macOS puede tardar en arrancar: tolerar lecturas fallidas
        if fallos < 60:
            fallos += 1
            cv.waitKey(50)
            continue
        print("No se pudieron leer frames (revisa el permiso de cámara).")
        break
    fallos = 0

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    for (x, y, w, h) in rostros:
        roi = cv.resize(gray[y:y + h, x:x + w], TAM, interpolation=cv.INTER_CUBIC)
        roi = cv.equalizeHist(roi)  # igual que en entrenar_eigen.py
        label_id, confidence = face_recognizer.predict(roi)

        if confidence < args.umbral:
            color = (0, 255, 0)
            text = f"{faces[label_id]} ({confidence:.0f})"
        else:
            color = (0, 0, 255)
            # Mostrar a quién se parece más ayuda a calibrar el umbral
            text = f"Desconocido ~{faces[label_id]} ({confidence:.0f})"

        cv.putText(frame, text, (x, y - 12), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2, cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv.imshow("Eigenfaces (ESC para salir)", frame)
    if cv.waitKey(1) == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
