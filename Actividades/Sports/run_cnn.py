"""Corre la CNN de sportimages con epocas y capas convolucionales configurables.

Uso (desde la carpeta Sports):
    python run_cnn.py --name exp1 --epochs 10 --conv 1

Guarda en resultados/<name>/ la matriz de confusion (png y csv), las curvas de
entrenamiento, el reporte por clase y agrega una fila a resultados/resumen.csv.
"""
import argparse, csv, os, re, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="nombre del experimento, ej. exp1")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--conv", type=int, default=1, choices=[1, 2, 3], help="bloques convolucionales")
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, LeakyReLU, Input)

tf.keras.utils.set_random_seed(a.seed)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "sportimages")
CACHE = os.path.join(BASE, "sportimages_cache.npz")
OUT = os.path.join(BASE, "resultados", a.name)
os.makedirs(OUT, exist_ok=True)

# ---- datos (la primera vez lee las 77k imagenes y guarda un cache) ----
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    X, y, deportes = z["X"], z["y"], list(z["deportes"])
    print("Cache cargado:", X.shape)
else:
    deportes = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    imgs, labels = [], []
    for idx, d in enumerate(deportes):
        for f in sorted(os.listdir(os.path.join(DATA, d))):
            if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", f, re.I):
                im = plt.imread(os.path.join(DATA, d, f))
                if im.ndim == 3 and im.shape == (21, 28, 3):
                    imgs.append(im); labels.append(idx)
        print("leido", d, len(labels))
    X = np.array(imgs, dtype=np.uint8); y = np.array(labels)
    np.savez_compressed(CACHE, X=X, y=y, deportes=np.array(deportes))
nClasses = len(deportes)
print("Clases:", dict(enumerate(deportes)), "| imagenes:", len(X))

# mismos splits en todos los experimentos (misma semilla)
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=a.seed)
train_X = train_X.astype("float32") / 255.; test_X = test_X.astype("float32") / 255.
train_Y1, test_Y1 = to_categorical(train_Y, nClasses), to_categorical(test_Y, nClasses)
train_X, valid_X, train_L, valid_L = train_test_split(train_X, train_Y1, test_size=0.2, random_state=13)

# ---- modelo: con --conv 1 es el de la practica ----
m = Sequential([Input(shape=(21, 28, 3))])
for i in range(a.conv):
    m.add(Conv2D(32 * 2 ** i, (3, 3), activation="linear", padding="same"))
    m.add(LeakyReLU(negative_slope=0.1))
    m.add(MaxPooling2D((2, 2), padding="same"))
    m.add(Dropout(0.5))
m.add(Flatten())
m.add(Dense(32, activation="linear")); m.add(LeakyReLU(negative_slope=0.1)); m.add(Dropout(0.5))
m.add(Dense(nClasses, activation="softmax"))
m.summary()
m.compile(loss="categorical_crossentropy",
          optimizer=tf.keras.optimizers.SGD(learning_rate=a.lr), metrics=["accuracy"])

t0 = time.time()
h = m.fit(train_X, train_L, batch_size=a.batch, epochs=a.epochs, verbose=1,
          validation_data=(valid_X, valid_L))
seg = time.time() - t0

# ---- evaluacion ----
loss, acc = m.evaluate(test_X, test_Y1, verbose=0)
pred = np.argmax(m.predict(test_X, verbose=0), axis=1)
cm = confusion_matrix(test_Y, pred)
rep = classification_report(test_Y, pred, target_names=deportes, digits=3)
rep_d = classification_report(test_Y, pred, target_names=deportes, output_dict=True)
print(rep)

np.savetxt(os.path.join(OUT, "matriz_confusion.csv"), cm, fmt="%d", delimiter=",",
           header=",".join(deportes), comments="")
with open(os.path.join(OUT, "reporte.txt"), "w") as f:
    f.write(f"{a.name}: epocas={a.epochs} conv={a.conv} batch={a.batch} lr={a.lr}\n")
    f.write(f"test accuracy={acc:.4f} test loss={loss:.4f} tiempo={seg:.0f}s params={m.count_params()}\n\n")
    f.write(rep)

fig, ax = plt.subplots(1, 2, figsize=(18, 8))
ConfusionMatrixDisplay(cm, display_labels=deportes).plot(ax=ax[0], xticks_rotation=45, colorbar=False)
ax[0].set_title("Conteo (filas = real, columnas = predicho)")
ConfusionMatrixDisplay(confusion_matrix(test_Y, pred, normalize="true"), display_labels=deportes
                       ).plot(ax=ax[1], xticks_rotation=45, colorbar=False, values_format=".2f")
ax[1].set_title("Normalizada por fila (= recall)")
fig.suptitle(f"{a.name}: {a.epochs} epocas, {a.conv} capa(s) conv, acc={acc:.3f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "matriz_confusion.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h.history["accuracy"], label="train"); ax[0].plot(h.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].legend()
ax[1].plot(h.history["loss"], label="train"); ax[1].plot(h.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "curvas.png"), dpi=120); plt.close(fig)

m.save(os.path.join(OUT, "modelo.keras"))

resumen = os.path.join(BASE, "resultados", "resumen.csv")
nuevo = not os.path.exists(resumen)
with open(resumen, "a", newline="") as f:
    w = csv.writer(f)
    if nuevo:
        w.writerow(["experimento", "epocas", "capas_conv", "parametros", "test_acc",
                    "test_loss", "macro_f1", "segundos"])
    w.writerow([a.name, a.epochs, a.conv, m.count_params(), f"{acc:.4f}", f"{loss:.4f}",
                f"{rep_d['macro avg']['f1-score']:.4f}", f"{seg:.0f}"])
print(f"\nListo. Resultados en {OUT}\nAccuracy test: {acc:.4f}")
