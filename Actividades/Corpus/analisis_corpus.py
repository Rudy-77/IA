"""Analisis del corpus de comentarios "no todos pueden votar".

Contrasta lo que AFIRMAN los comentarios contra las columnas del dataset
(puede_votar, cuenta_verificada, dias_cuenta, voto) y marca lo que los datos
no pueden sustentar.  Uso:  python analisis_corpus.py
Salidas (carpeta resultados/): afirmaciones.csv, resumen.md, graficas .png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).parent
OUT = AQUI / "resultados"
OUT.mkdir(exist_ok=True)

d = pd.read_csv(AQUI / "comentarios_no_todos_pueden_votar.csv")


def base(t):
    """Normaliza un comentario: quita signos, hashtags y muletillas."""
    t = t.lower()
    t = re.sub(r"#\w+|[.?!]+", "", t)
    t = re.sub(r"\b(pls|ya)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


d["afirmacion"] = d.comentario.map(base)
# Algunos comentarios vienen cortados por la limpieza ("...manipulan la"); se
# unen al texto completo mas largo que los contiene.
unicas = sorted(d.afirmacion.unique(), key=len, reverse=True)
def completa(t):
    for u in unicas:
        if u.startswith(t):
            return u
    return t
d["afirmacion"] = d.afirmacion.map(completa)

# ---------- 1. Calidad del corpus ----------
cols_dup = ["comentario", "categoria", "puede_votar", "cuenta_verificada", "dias_cuenta", "voto"]
calidad = {
    "filas": len(d),
    "textos_distintos": d.comentario.nunique(),
    "afirmaciones_unicas": d.afirmacion.nunique(),
    "filas_duplicadas": int(d.duplicated(cols_dup).sum()),
    "nulos": int(d.isna().sum().sum()),
}

# ---------- 2. Regla de dias de cuenta ----------
pv = d[d.puede_votar == 1]
npv = d[d.puede_votar == 0]
UMBRAL = 30
excep_a = d[(d.puede_votar == 1) & (d.dias_cuenta <= UMBRAL)]   # pueden con <=30 dias
excep_b = d[(d.puede_votar == 0) & (d.dias_cuenta > UMBRAL)]    # no pueden con >30 dias
cumple_dias = 1 - (len(excep_a) + len(excep_b)) / len(d)

# ---------- 3. Verificacion ----------
verif = d[d.cuenta_verificada == 1]
verif_pueden = verif.puede_votar.mean()
pueden_no_verif = int(((d.puede_votar == 1) & (d.cuenta_verificada == 0)).sum())
no_pueden_verif = int(((d.puede_votar == 0) & (d.cuenta_verificada == 1)).sum())

# ---------- 4. Coherencia postura <-> voto ----------
af = d[d.categoria == "a_favor_restriccion"].voto.value_counts()
ec = d[d.categoria == "en_contra_restriccion"].voto.value_counts()
coherencia = {
    "a_favor_vota_abrir": int(af.get("abrir_voto_a_todos", 0)),
    "en_contra_vota_restringir": int(ec.get("a_favor_de_restringir", 0)),
}
quejas_sin_permiso = (d[d.categoria == "queja"].puede_votar == 0).mean()

# ---------- 5. Tabla de afirmaciones -> evidencia ----------
def evalua(a, sub):
    """Devuelve (variable, metrica, veredicto) para una afirmacion."""
    n = len(sub)
    if sub.categoria.iloc[0] == "neutro_pregunta":
        return ("-", "pregunta, no es una afirmacion", "NO APLICA")
    if "30 días" in a:
        return ("dias_cuenta/puede_votar", f"{cumple_dias:.1%} de las filas cumplen el umbral de {UMBRAL} dias", "SUSTENTADA")
    if "recién creadas" in a or "cuentas nuevas" in a or "poco tiempo registrado" in a:
        m = (sub.dias_cuenta <= 32).mean()
        return ("dias_cuenta", f"{m:.1%} de quienes lo dicen tienen <=32 dias" if "poco tiempo" in a
                else f"los bloqueados tienen mediana {npv.dias_cuenta.median():.0f} dias vs {pv.dias_cuenta.median():.0f}", "SUSTENTADA")
    if "verificad" in a:
        return ("cuenta_verificada", f"verificadas->pueden {verif_pueden:.0%}, pero {pueden_no_verif} no verificadas tambien pueden", "PARCIAL")
    if re.search(r"\bbots?\b", a) or "falsos" in a or "manipul" in a:
        return ("(no existe)", "no hay columna que identifique bots", "SIN EVIDENCIA EN EL DATASET")
    if any(k in a for k in ["alumno", "staff", "institucional", "inscripción", "nuevo ingreso"]):
        return ("(no existe)", "no hay columna de rol/alumno/staff", "SIN EVIDENCIA EN EL DATASET")
    if "403" in a or "bloquea" in a or "rechaza" in a or "permiso" in a or "no tienes" in a:
        return ("puede_votar", f"{(sub.puede_votar == 0).mean():.0%} de quienes lo dicen tienen puede_votar=0", "SUSTENTADA")
    if "válida" in a or "democr" in a or "discrimina" in a or "legítima" in a or "justo" in a or "silencia" in a or "para qué preguntan" in a:
        return ("(juicio de valor)", "opinion; los datos no la confirman ni la niegan", "OPINION")
    if "reglamento" in a or "moderación" in a or "sospechoso" in a or "sin verificación" in a:
        return ("(requiere reglamento)", "norma institucional; falta el documento fuente", "FALTA FUENTE EXTERNA")
    if sub.categoria.iloc[0] == "a_favor_restriccion":
        return ("(juicio de valor)", "opinion; los datos no la confirman ni la niegan", "OPINION")
    return ("-", "pregunta o peticion, no es una afirmacion", "NO APLICA")

filas = []
for a, sub in d.groupby("afirmacion"):
    var, met, ver = evalua(a, sub)
    filas.append({
        "afirmacion": a,
        "categoria": sub.categoria.iloc[0],
        "n_comentarios": len(sub),
        "variable_que_la_prueba": var,
        "evidencia": met,
        "veredicto": ver,
    })
tabla = pd.DataFrame(filas).sort_values(["veredicto", "n_comentarios"], ascending=[True, False])
tabla.to_csv(OUT / "afirmaciones.csv", index=False, encoding="utf-8-sig")
por_veredicto = tabla.groupby("veredicto").n_comentarios.agg(["count", "sum"]).rename(
    columns={"count": "afirmaciones", "sum": "comentarios"})

# ---------- 6. Graficas ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist([npv.dias_cuenta, pv.dias_cuenta], bins=range(0, 1800, 10),
        label=["No puede votar", "Puede votar"], color=["#c0392b", "#2e86c1"], stacked=True)
ax.axvline(UMBRAL, color="k", ls="--", lw=1); ax.text(UMBRAL + 15, 300, "30 dias")
ax.set_yscale("log"); ax.set_xlabel("Dias de cuenta"); ax.set_ylabel("Comentarios (escala log)"); ax.legend(loc="upper right")
ax.set_title("Antiguedad de cuenta vs permiso de voto")
fig.tight_layout(); fig.savefig(OUT / "dias_vs_permiso.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
pd.crosstab(d.categoria, d.voto).plot.barh(stacked=True, ax=ax,
    color=["#27ae60", "#2e86c1", "#95a5a6", "#c0392b"])
ax.set_title("Postura del comentario vs voto emitido"); ax.set_xlabel("Comentarios"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "categoria_vs_voto.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8))
por_veredicto.comentarios.sort_values().plot.barh(ax=ax, color="#5d6d7e")
ax.set_title("Comentarios segun el respaldo en el dataset"); ax.set_xlabel("Comentarios"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "veredictos.png", dpi=150); plt.close(fig)

# ---------- 7. Resumen ----------
EXPLICA = {
    "SUSTENTADA": "Los datos la confirman (se cumple en casi todas las filas).",
    "PARCIAL": "Los datos la confirman a medias: hay casos que la contradicen.",
    "SIN EVIDENCIA EN EL DATASET": "Habla de algo que el dataset no mide (bots, rol de alumno/staff...).",
    "FALTA FUENTE EXTERNA": "Es una norma institucional; hace falta el documento oficial en el corpus.",
    "OPINION": "Juicio de valor; los datos no lo pueden confirmar ni negar.",
    "NO APLICA": "Es una pregunta o una peticion, no afirma nada que se pueda comprobar.",
}
filas_ver = "\n".join(
    f"| {v} | {int(r.afirmaciones)} | {int(r.comentarios)} | {EXPLICA[v]} |"
    for v, r in por_veredicto.sort_values("comentarios", ascending=False).iterrows())

md = f"""# Analisis del corpus: "no todos pueden votar"

**Objetivo:** revisar que tan bien sustenta el dataset lo que dicen los comentarios.
Cada comentario *afirma* algo ("necesitas 30 dias", "solo los verificados votan",
"hay bots"...). Aqui se compara cada afirmacion contra los datos reales de la cuenta
(`puede_votar`, `cuenta_verificada`, `dias_cuenta`, `voto`).

## Que significa cada columna
| Columna | Significado |
|---|---|
| `comentario` | Texto que escribio el usuario |
| `categoria` | Tipo de comentario: queja, explica_regla, a_favor_restriccion, en_contra_restriccion, neutro_pregunta |
| `sentimiento` | Positivo / neutro / negativo (equivale 1 a 1 a `categoria`) |
| `puede_votar` | 1 si la cuenta tiene permiso de voto, 0 si no |
| `cuenta_verificada` | 1 si la cuenta esta verificada |
| `dias_cuenta` | Antiguedad de la cuenta en dias |
| `voto` | Lo que voto: a_favor_de_restringir, abrir_voto_a_todos, abstencion o sin_permiso |

## 1. Calidad del corpus
*Que es:* revisar si el dataset es confiable antes de sacar conclusiones.

- Filas: **{calidad['filas']}**, valores nulos: **{calidad['nulos']}**.
- Textos distintos: **{calidad['textos_distintos']}**, pero al quitar signos, hashtags y muletillas ("pls", "??") quedan solo **{calidad['afirmaciones_unicas']} afirmaciones unicas**. El corpus repite plantillas con pequenas variaciones.
- Filas duplicadas (mismo comentario y mismos datos, sin contar id ni usuario): **{calidad['filas_duplicadas']}**.
- `categoria` y `sentimiento` dicen lo mismo, asi que una de las dos sobra al modelar.

*Por que importa:* con tan poca variedad, un modelo solo memorizaria las frases.

## 2. Regla de dias de cuenta
*Que es:* comprobar si es cierto que hace falta una antiguedad minima para votar. El histograma muestra
cuantos comentarios hay por dias de cuenta, en rojo los que **no** pueden votar y en azul los que **si**
(la escala es logaritmica para que se vea el detalle).

![Antiguedad de cuenta vs permiso de voto](dias_vs_permiso.png)

- No pueden votar: **{len(npv)}** cuentas, con **{npv.dias_cuenta.min()} a {npv.dias_cuenta.max()}** dias (mediana {npv.dias_cuenta.median():.0f}).
- Si pueden votar: **{len(pv)}** cuentas, con mediana de **{pv.dias_cuenta.median():.0f}** dias.
- Con un umbral de **{UMBRAL} dias** la regla se cumple en **{cumple_dias:.1%}** de las filas.
  Excepciones: **{len(excep_a)}** cuentas pueden votar con {UMBRAL} dias o menos y **{len(excep_b)}** no pueden con mas de {UMBRAL}.

*Conclusion:* los comentarios que dicen "necesitas mas de 30 dias" o "las cuentas nuevas no pueden votar" **estan sustentados**.

## 3. Verificacion de cuenta
*Que es:* comprobar si "solo las cuentas verificadas pueden votar".

- Cuentas verificadas: **{len(verif)}**; de ellas pueden votar el **{verif_pueden:.0%}**. Verificadas que no pueden: **{no_pueden_verif}**.
- Cuentas **no** verificadas que tambien pueden votar: **{pueden_no_verif}**.

*Conclusion:* estar verificado **basta** para votar, pero **no es requisito**. La afirmacion es solo **parcialmente cierta**; pesa mas la antiguedad.

## 4. Coherencia entre postura y voto
*Que es:* ver si lo que la gente opina coincide con lo que vota. La grafica cuenta, por tipo de comentario, que voto emitieron.

![Postura del comentario vs voto emitido](categoria_vs_voto.png)

- Comentarios a favor de restringir que votaron "abrir a todos": **{coherencia['a_favor_vota_abrir']}**.
- Comentarios en contra de restringir que votaron "a favor de restringir": **{coherencia['en_contra_vota_restringir']}**.
- Quejas hechas por cuentas sin permiso: **{quejas_sin_permiso:.0%}**.

*Conclusion:* las posturas son coherentes con el voto y todas las quejas vienen de cuentas realmente bloqueadas.
Los comentarios neutros (explica_regla, neutro_pregunta) se reparten entre todos los votos, como es de esperar.

## 5. Respaldo de cada afirmacion
*Que es:* se agruparon los comentarios en afirmaciones unicas y a cada una se le asigno un **veredicto** segun
si los datos la pueden comprobar. El detalle esta en `afirmaciones.csv`.

![Comentarios segun el respaldo en el dataset](veredictos.png)

| Veredicto | Afirmaciones | Comentarios | Que significa |
|---|---|---|---|
{filas_ver}

**Nota:** la clasificacion usa reglas de palabras clave, asi que conviene revisar `afirmaciones.csv` por si alguna quedo mal puesta.

## Pendiente: fuentes externas para el corpus
- Reglamento o reglas oficiales (30 dias, verificacion, correo institucional, moderacion).
- Evidencia sobre bots y manipulacion de encuestas en foros.
- Columnas nuevas ideales: `es_bot`, `rol` (alumno/staff) y un registro de intentos de voto.
"""
(OUT / "resumen.md").write_text(md, encoding="utf-8")
print(md)
print(tabla.to_string(index=False))
