# Reporte EDA — Operación Cuatro Frentes
**Analista:** Paulo César Pérez Martínez

> Todo el análisis sale de las muestras, los resúmenes y las tablas del enunciado. Donde digo "en la muestra" hablo solo de esas 10 o 12 filas, que no son representativas de la población completa.

---

## Misión 1 — Semáforo Académico

- **Pregunta de negocio:** ¿qué nivel de riesgo de reprobar (verde, amarillo o rojo) tiene cada alumno al cierre del parcial?
- **Tipo propuesto: clasificar** (multiclase, tres clases con orden natural). **Justificación:** `riesgo` es una etiqueta de categoría, no una cantidad; la pregunta es "¿a qué grupo pertenece?". No es binaria porque hay tres niveles.
- **Y / forma de Y:** `riesgo`, categórica ordinal (verde < amarillo < rojo).
- **X (mínimo 5):**
  - `asistencia_pct`: quien falta mucho pierde contenido y suele reprobar.
  - `promedio_parciales`: es la evidencia directa del rendimiento hasta ahora.
  - `tareas_entregadas`: mide constancia y compromiso.
  - `reprobadas_previas`: el historial es un buen indicador de riesgo.
  - `horas_plataforma`: refleja el esfuerzo de estudio fuera del aula.
  - `turno`: puede aportar contexto, pero en la muestra no parece separar grupos.
- **Hallazgos EDA (dos patrones claros):**
  1. **El riesgo sube cuando bajan asistencia y promedio.** Medias: verde 91 % de asistencia y promedio 8,4; amarillo 76 % y 6,8; rojo 52 % y 5,0. En la muestra, los cuatro rojos tienen asistencia ≤ 60 y promedio ≤ 5,5, y los cuatro verdes tienen asistencia ≥ 88 y promedio ≥ 7,8.
  2. **Las materias reprobadas previas crecen con el riesgo:** 0,2 en verde, 0,9 en amarillo y 2,4 en rojo (en la muestra, todos los rojos tienen 2 o más). Pero la frontera verde/amarillo se traslapa: el alumno 109 (80 %, promedio 7,2, 0 reprobadas) es amarillo mientras el 102 (88 %, 7,8, 0 reprobadas) es verde. Esto indica que ninguna variable separa sola los grupos.
  - `turno` no aporta señal visible: hay matutinos y vespertinos en los tres colores.
- **Distribución de Y:** verde 120 (40 %), amarillo 105 (35 %), rojo 75 (25 %), con N = 300. Está razonablemente balanceada, sin una clase minúscula. Aun así, rojo es la minoría y la más costosa de fallar, así que no basta con la accuracy: un modelo que siempre diga "verde" ya acertaría el 40 %. Hay que reportar F1 macro, recall de la clase roja y la matriz de confusión (confundir rojo con verde es peor que confundir rojo con amarillo).
- **Calidad de datos (problema inventado y cómo detectarlo):** `horas_plataforma` con valores faltantes en alumnos que nunca entraron al LMS (en realidad son 0, no "desconocido"). Se detecta con `isna().sum()` por columna y cruzando con `tareas_entregadas`; también reviso rangos (`asistencia_pct` fuera de 0–100 o `promedio_parciales` fuera de 0–10) con `describe()` y mínimos y máximos.
- **Leakage evitado:** **no** usaría la calificación final del curso como X. Se conoce hasta el final del semestre y es justo el resultado que el semáforo intenta anticipar; el modelo sacaría métricas casi perfectas en validación y no serviría al cierre del parcial. Lo mismo aplica a cualquier columna calculada después del parcial.
- **Métricas coherentes con mi tipo:** F1 macro, recall de la clase rojo, matriz de confusión (y, de forma opcional, kappa ponderado por ser ordinal).
- **Modelo propuesto y condición:** árbol de decisión poco profundo (también sirve Random Forest). Condición: N = 300 es pequeño, así que hay que limitar la profundidad (3 a 4) y validar con validación cruzada estratificada para que cada clase, sobre todo rojo (75 casos), esté representada en cada partición.

---

## Misión 2 — Alerta de Churn

- **Pregunta de negocio:** ¿abandonará el alumno la materia, dada su actividad a mitad de semestre?
- **Tipo propuesto: clasificar** (binaria). **Justificación:** `abandona` solo toma los valores 0 y 1, que son etiquetas ("abandona" / "permanece"), no una cantidad. Aunque estén escritas como números, no se pueden promediar con sentido.
- **Distribución de Y e implicaciones:** con N = 500 y 14 %, hay **500 × 0,14 = 70 positivos** y 430 negativos (razón de unos 1 : 6). Si alguien reporta "86 % de aciertos" sin contexto, ese número es exactamente lo que logra un modelo que responde "nadie abandona": acierta los 430 que permanecen y falla los 70 que abandonan. Es decir, 86 % de accuracy con 0 % de recall en la clase que importa. Por eso la accuracy engaña.
- **Tratamiento de NA (`calif_actividad_1`):**
  - **Hipótesis:** los faltantes **no son aleatorios**. En la muestra, los cuatro alumnos con NA (203, 204, 208, 210) abandonan los cuatro; entre los 6 que tienen nota, solo 1 abandona. Lo más probable es que NA signifique "no entregó la actividad".
  - **Tratamiento:** no borrar esas filas (se perdería la señal) y usar **las dos cosas**: una bandera `tiene_calif` (0/1) y una imputación del valor (por ejemplo, la mediana o 0, según se justifique). Hay que revisar el momento: la bandera solo es válida si el NA se conoce antes de la fecha de predicción (mitad de semestre). Si el NA aparece porque el alumno ya había abandonado, sería una fuga de información.
- **Métricas y costo de error:** como el costo no es simétrico, el error grave es el **falso negativo** (no detectar a quien va a abandonar: se pierde la oportunidad de intervenir). El falso positivo cuesta poco (contactar a un alumno que no lo necesitaba). Entonces priorizo **recall de la clase abandona = 1**, cuidando la precision, y reporto F1, PR-AUC y la matriz de confusión; el umbral de decisión se baja de 0,5 si hace falta.
- **Leakage:** **no** incluiría la "fecha de baja definitiva" (es el propio resultado, o su consecuencia directa) ni la "nota final" (existe después de que termina el curso). Ambas contienen el futuro, y el modelo se vería perfecto en validación sin servir para predecir a mitad de semestre.
- **Tres preguntas EDA respondidas con los datos:**
  1. *¿Cómo está distribuida Y?* 86 % / 14 % (430 / 70): fuerte desbalance.
  2. *¿Hay outliers en `dias_sin_login`?* En la muestra los valores son 1, 2, 3, 4, 6, 14, 18, 21, 25 y 30: cola larga a la derecha. Los valores altos no son errores, **son la señal**: los cinco alumnos que abandonan tienen 14 días o más sin entrar, y ninguno de los que permanecen pasa de 6. Se conservan (si acaso, una transformación logarítmica).
  3. *¿Abandonan más los que trabajan?* En la muestra, 4 de 6 alumnos que trabajan abandonan (67 %) contra 1 de 4 entre los que no trabajan (25 %): hay relación, pero el alumno 208 abandona sin trabajar, así que `trabaja` no basta. Coincide con la fuerza "débil–moderada" del enunciado. Nota: la muestra tiene 50 % de abandono (5 de 10) frente al 14 % real, así que sirve para ver patrones, no para estimar tasas.
  - Colinealidad sospechosa: `dias_sin_login` alto y `avance_contenido_pct` bajo aparecen juntos (18 días → 15 %; 30 días → 0 %); hay que revisar la matriz de correlación, sobre todo si se usa regresión logística.
- **Comparación con la Misión 1:** las dos son clasificación, pero la forma de Y cambia. En la M1 hay tres clases con balance razonable (40/35/25) y el EDA se centra en comparar las medias por clase; en la M2 hay solo dos clases y muy desbalanceadas (86/14), además de NA informativos y un riesgo de leakage por la fecha del evento. Por eso las métricas también cambian: F1 macro en la M1, recall y precision de la clase positiva en la M2.
- **Modelo propuesto:** regresión logística o Random Forest con `class_weight='balanced'`, evaluado con recall, precision, F1 y PR-AUC, y con el umbral ajustado al costo de no detectar un abandono.

---

## Misión 3 — Pronóstico de Puntaje

- **Pregunta de negocio:** ¿qué calificación final (de 0 a 100) obtendrá el alumno?
- **Tipo propuesto: predecir (regresión).** **Justificación:** `calificacion_final` es un valor numérico continuo en una escala de 0 a 100; la pregunta es "¿qué número espera?", y las diferencias entre valores tienen sentido (un 89 está más cerca de un 90 que de un 40).
- **Y y X:** `Y = calificacion_final`. `X`: `promedio_tareas`, `examen_1`, `examen_2`, `asistencia_pct` y `horas_estudio_sem`. Según las pistas, la más relacionada es `examen_1` y `examen_2` (correlación ~0,85 con la calificación final). Importante: `examen_2` solo es válida como X si el pronóstico se hace después de ese examen.
- **Convertir a aprobado/reprobado:**
  - *Se gana:* un problema más simple de explicar y de accionar (alertas, listas de riesgo), con métricas conocidas de clasificación.
  - *Se pierde:* información. Un 61 y un 99 pasan a ser lo mismo, y un 59 y un 61 se tratan como opuestos aunque casi no difieren. Ya no se puede saber cuántos puntos le faltan a un alumno, ni ordenar a los alumnos por rendimiento esperado. Además, el umbral (por ejemplo 60) es una decisión arbitraria que se vuelve parte del problema.
- **Media vs mediana de `horas_estudio_sem`:** media 5,5 y mediana 5, con desviación estándar 3,2 y máximo 25. La diferencia entre media y mediana es pequeña (0,5), pero el máximo está a (25 − 5,5) / 3,2 ≈ 6 desviaciones de la media: hay una **cola derecha** con valores de 20 a 25 h, probablemente exageraciones del auto-reporte. No eliminaría esos valores sin revisar: los verificaría, y de no ser creíbles los limitaría (winsorizar) o usaría una transformación logarítmica.
- **Filas con `calificacion_final > 100`:** son errores de captura, porque la escala llega a 100. No se pueden dejar tal cual. Primero se comprueba contra la fuente original; si se puede corregir (un error de tecleo), se corrige; si no, se marcan como NA y se excluyen o imputan, documentando cuántas fueron (3 de 200). Por cierto, el resumen muestra un máximo de 99, lo que sugiere que ya fue calculado sin esas 3 filas.
- **Dos métricas coherentes:**
  - **MAE** (error absoluto medio): "en promedio me equivoco X puntos"; se interpreta directo en la escala 0–100.
  - **RMSE** (raíz del error cuadrático medio): castiga más los errores grandes, útil para detectar casos donde el modelo falla por mucho. Opcionalmente, **R²** para ver qué parte de la variación se explica.
- **Qué modelo encaja:** si al graficar `examen_1` contra `calificacion_final` sale casi una línea, basta una **regresión lineal** simple (o múltiple). Si la relación fuera en "escalones", se ajustaría mejor un **árbol de regresión**, que predice un valor constante por tramo.
- **Comparación con la Misión 1:** misma escuela, pero otra pregunta. En la M1 `Y` es una etiqueta (riesgo) y el EDA compara grupos y balance de clases; en la M3 `Y` es un número y el EDA se centra en la distribución, los outliers, los rangos válidos y las correlaciones. Las métricas pasan de F1 y recall a MAE y RMSE. Además, las dos preguntas están relacionadas: el riesgo de la M1 podría derivarse de la nota de la M3, pero al hacerlo se pierde precisión.
- **Modelo propuesto:** regresión lineal múltiple como línea base; árbol de regresión o Random Forest regresor si la relación resulta no lineal.

---

## Misión 4 — Tiempo de Estudio

- **Pregunta de negocio:** ¿cuántas horas adicionales necesita un alumno para dominar el tema?
- **Tipo propuesto: predecir (regresión).** **Justificación:** `horas_adicionales` es un valor numérico continuo (2,0, 5,5, 18,0, 35,0...) y la respuesta es una cantidad, no una categoría.
- **Hipótesis EDA (dificultad → horas):** a mayor dificultad del tema, más horas adicionales **y también más dispersión**. Las medias son 3,2 (baja), 7,8 (media) y 16,5 (alta), y las medianas 2,5, 7,0 y 14,0; en cada nivel la media supera a la mediana, señal de cola derecha.
- **Cómo inspeccionar las categóricas (`tema_dificultad` y `dispositivo`):** con `value_counts()` para ver el balance (baja 80, media 90, alta 70 de 240), con `describe()` o un boxplot de `horas_adicionales` por categoría, y con una tabla cruzada de dificultad × dispositivo. Esto último es clave: en la muestra, los alumnos en móvil tienen muchas horas (18, 28, 12,5 y 35), pero tres de esos cuatro están en temas de dificultad alta, así que el efecto del dispositivo puede ser solo un efecto de confusión; hay que compararlo dentro de cada nivel de dificultad. También reviso errores de escritura en las categorías.
- **Cola > 40 h (3 % de 240, o sea unos 7 alumnos): ¿borrar o conservar?** Conservar, salvo que se compruebe que son errores. Primero verifico que sean coherentes (por ejemplo, dificultad alta y `pretest_score` bajo). Si es así, son los alumnos que más tutoría necesitan, y borrarlos haría que el modelo subestime justo a quienes más importan. Si parecen errores (un 400 en vez de un 40), se corrigen o se descartan documentándolo. Para entrenar con ellos, uso una transformación logarítmica de `Y` o un modelo robusto a valores extremos.
- **Redundancia entre `pretest_score` y `ejercicios_correctos_pct`:** en la muestra se mueven casi 1 a 1 (85/90, 40/45, 92/95, 25/30, 60/60...), así que sospecho una correlación muy alta. Lo compruebo con la matriz de correlación, un diagrama de dispersión y el VIF. Si es redundante, conservo una de las dos o las combino; en regresión lineal duplicarlas vuelve inestables los coeficientes.
- **Alternativa: binarizar `Y` con un umbral de 15 h ("tutoría intensiva: sí/no").** Tiene sentido si la decisión real es binaria (hay cupo limitado de tutoría intensiva y el umbral viene de esa política). Se pierde la magnitud (18 h y 35 h quedan iguales, 14 h y 16 h quedan en lados opuestos) y el resultado es sensible al umbral. Con la distribución dada, la clase positiva (más de 15 h) estaría entre el 13 % (las filas de 20 h o más) y el 35 %, y cerca del 24 % si se reparte de forma uniforme dentro del tramo de 10 a 20 h, así que además habría cierto desbalance.
- **Métricas y modelo, con 2 chequeos EDA obligatorios:**
  - **Métricas:** MAE (en horas, fácil de comunicar a tutoría) y RMSE (castiga los fallos grandes de la cola). Evitaría el error porcentual porque con valores de 1 o 2 horas se dispara.
  - **Modelo:** regresión lineal sobre `log(horas_adicionales)` como línea base, y árbol de regresión o Random Forest regresor para capturar la interacción entre dificultad y puntaje.
  - **Chequeos antes de entrenar:** (1) revisar la redundancia entre `pretest_score` y `ejercicios_correctos_pct`; (2) confirmar si la cola > 40 h es real o error de captura, y revisar la distribución de `Y` por nivel de dificultad.

---

## Síntesis (máx. 8 líneas)

1. **Mis cuatro propuestas:**

| Misión | Y | Tipo propuesto |
|---|---|---|
| M1 Semáforo Académico | `riesgo` (verde/amarillo/rojo) | Clasificar (multiclase ordinal) |
| M2 Alerta de Churn | `abandona` (0/1) | Clasificar (binaria, desbalanceada) |
| M3 Pronóstico de Puntaje | `calificacion_final` (0–100) | Predecir (regresión) |
| M4 Tiempo de Estudio | `horas_adicionales` (horas) | Predecir (regresión) |

2. **Pista que usé para decidir "clase vs número":** si los valores de Y se pueden promediar con sentido y la diferencia entre dos valores significa algo, es un número (M3 y M4); si son etiquetas, aunque aparezcan como 0 y 1 (M2), es una clase.
3. **Frase final:** el tipo de problema se deduce de la pregunta y de Y porque la pregunta define si queremos un grupo o una cantidad, y la forma de Y define qué métricas, qué EDA y qué modelos tienen sentido.
