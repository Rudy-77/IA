"""Captura de rostros para el dataset.

Uso:
    python captura.py persona1 persona1.mp4          # desde un video
    python captura.py persona1 0                     # desde la webcam
    python captura.py persona1 0 --max 1000 --salto 3

Guarda recortes de rostro en escala de grises de 100x100 en dataset/<nombre>/.
Si hay varios rostros en un frame, solo guarda el más grande (así no se
mezclan identidades en la misma carpeta).
"""
import argparse
import os

import cv2 as cv

TAM = (100, 100)

parser = argparse.ArgumentParser()
parser.add_argument("nombre", help="nombre de la persona (nombre de la carpeta)")
parser.add_argument("fuente", help="ruta del video o 0 para la webcam")
parser.add_argument("--max", type=int, default=1000, help="imágenes a guardar")
parser.add_argument("--salto", type=int, default=3,
                    help="guardar 1 de cada N frames (más variedad)")
parser.add_argument("--salida", default="dataset")
args = parser.parse_args()

carpeta = os.path.join(args.salida, args.nombre)
os.makedirs(carpeta, exist_ok=True)

# Continuar numeración si la carpeta ya tiene imágenes
count = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])
inicio = count

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

fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente
cap = cv.VideoCapture(fuente)
if not cap.isOpened():
    raise SystemExit(f"No se pudo abrir la fuente: {args.fuente}")

es_camara = isinstance(fuente, int)
frame_num = 0
frames_con_rostro = 0
fallos = 0
motivo = "se alcanzó el máximo de imágenes"
while count - inicio < args.max:
    ret, frame = cap.read()
    if not ret:
        # La cámara de macOS puede tardar en arrancar: tolerar lecturas fallidas
        if es_camara and fallos < 60:
            fallos += 1
            cv.waitKey(50)
            continue
        motivo = ("no se pudieron leer frames de la cámara (revisa el permiso de "
                  "cámara) " if es_camara else "terminó el video")
        break
    fallos = 0
    frame_num += 1

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    if len(rostros) > 0:
        frames_con_rostro += 1
        x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if frame_num % args.salto == 0:
            roi = cv.resize(gray[y:y + h, x:x + w], TAM,
                            interpolation=cv.INTER_CUBIC)
            cv.imwrite(os.path.join(carpeta, f"{args.nombre}_{count:04d}.jpg"), roi)
            count += 1

    cv.putText(frame, f"{args.nombre}: {count - inicio}/{args.max}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.imshow("captura (ESC para salir)", frame)
    if cv.waitKey(1) == 27:
        motivo = "presionaste ESC"
        break

cap.release()
cv.destroyAllWindows()
print(f"Terminó porque: {motivo}")
print(f"Frames leídos: {frame_num} | con rostro detectado: {frames_con_rostro}")
print(f"{count - inicio} imágenes nuevas guardadas en {carpeta} "
      f"(total en carpeta: {count})")
