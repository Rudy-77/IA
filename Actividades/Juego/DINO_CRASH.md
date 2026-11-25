# Reporte EDA — Operación Dino Crash
**Analista:** Paulo César Pérez Martínez


```
Datos crudos  →  EDA (entender)  →  Decisión de modelo  →  (luego) entrenar
```


## Bloque 1 — ¿Qué dataset necesitamos?

### Misión 1: Definir el problema y el dataset ideal

| ID | Pregunta | Una fila = |
|----|----------|------------|
| P1 | ¿Morirá en el siguiente frame? | Un frame de una partida |
| P2 | ¿Cuántos puntos alcanzará esta partida al morir? | Una partida completa |
| P3 | ¿Qué tipo de obstáculo viene próximo? | Un frame o evento |

---

### P1 — ¿Morirá en el siguiente frame?

**Variable objetivo (Y):** `morira_en_siguiente_frame` — **binaria** (0/1). No es numérica ni multiclase: es un evento puntual (colisión sí/no en el frame t+1).

**Variables de entrada (X)**:

| Variable | Por qué |
|---|---|
| `distancia_al_obstaculo_mas_cercano` (px) | Es la variable más directamente causal: a menor distancia con velocidad alta, mayor riesgo de colisión. |
| `velocidad_actual_juego` (px/s) | El juego acelera con el tiempo; la misma distancia es más peligrosa a mayor velocidad. |
| `tipo_obstaculo_proximo` (categórica: cactus_1/cactus_2/cactus_3/pajaro_bajo/pajaro_alto) | La hitbox y la acción correcta (saltar vs. agachar) dependen del tipo. |
| `estado_dino` (categórica: corriendo/saltando/agachado) | El dino es invulnerable en ciertas fases del salto; sin este dato no se puede saber si ya está esquivando. |
| `altura_actual_dino` o `velocidad_vertical_dino` | Determina si el salto en curso alcanzará a librar el obstáculo a tiempo. |
| `tiempo_desde_ultimo_input` (ms) | Un input tardío (reacción lenta) es un fuerte predictor de colisión inminente. |

**Granularidad:** frame a frame (cada ~16 ms a 60 fps). No sirve un resumen por partida: el evento que se predice ocurre a nivel de milisegundos, y la ventana útil de "anticipación" son unos pocos frames antes del choque, no la partida completa. Nota práctica: se puede *muestrear* más denso solo cerca de obstáculos y más disperso en tramos vacíos, pero no se puede saltar a granularidad de partida.

**Tamaño mínimo razonable — argumentado con números:**
Cada partida tiene, en el mejor de los casos, **un solo frame positivo** (el choque) contra miles de frames negativos. Ejemplo:
- Partida promedio: 30 s × 60 fps ≈ 1.800 frames.
- Si se define la clase positiva como "el frame exacto del choque" → proporción ≈ 1/1.800 (0,056%). Esto es un desbalance extremo.
- Si se define como "los 10 frames previos al choque son positivos" → proporción ≈ 10/1.800 (0,56%), sigue siendo muy desbalanceado.

Para tener al menos ~500 ejemplos positivos (mínimo razonable para que un clasificador aprenda el patrón sin sobreajustar a casos raros), y usando la definición de ventana de 10 frames, se necesitan **al menos 500 partidas completas**, y en la práctica conviene apuntar a 2.000–5.000 partidas para cubrir variedad de velocidades, tipos de obstáculo y estilos de juego.

**Riesgo si el dataset está mal definido:**
Si se etiqueta como positivo *solo* el frame exacto de la colisión (en vez de una ventana de frames previos), el modelo aprende a "detectar el choque cuando ya ocurrió", no a **anticiparlo**. Esto vuelve inútil el modelo para su propósito real (frenar/avisar antes de morir), porque en producción nunca se dispone del frame de colisión antes de que pase.

---

### P2 — ¿Cuántos puntos alcanzará esta partida al morir?

**Variable objetivo (Y):** `score_final` — **numérica continua** (en la práctica, entero creciente ligado al tiempo de supervivencia). Esto ya es una pista de diseño importante: como el score crece de forma casi monótona con el tiempo, `score_final` y `tiempo_de_supervivencia` están casi perfectamente correlacionados — hay que decidir cuál de las dos es realmente la variable de negocio antes de modelar, para no tratar dos veces la misma señal como si fueran independientes.

**Variables de entrada (X)**:

| Variable | Por qué |
|---|---|
| `velocidad_promedio_partida` | Partidas con velocidad de despegue más alta tienden a acortar el tiempo de reacción disponible. |
| `densidad_obstaculos` (obstáculos/segundo) | Más obstáculos por segundo = más oportunidades de error. |
| `proporcion_pajaros_vs_cactus` | Los pájaros exigen agachar, un input distinto; una racha de pájaros es un patrón de riesgo diferente. |
| `num_saltos_totales` / `num_agachadas_totales` | Actividad e involucramiento del jugador. |
| `tiempo_reaccion_promedio` (reaccion entre aparición del obstáculo y el input del jugador) | Habilidad/skill del jugador. |
| `jugador_id` o `nivel_habilidad` (si hay múltiples jugadores) | Sin esto, se analizaría un solo jugador. |

**Granularidad:** una fila = una partida completa, con variables **agregadas** desde el stream de frames (promedios, conteos, proporciones). No se puede reusar el dataset frame-a-frame de P1 sin antes agregarlo.

**Tamaño mínimo razonable:** con ~6–10 variables de entrada, una heurística estándar en regresión es pedir 10–20 observaciones por variable para evitar sobreajuste → **mínimo 100–200 partidas**, pero como el juego usa generación aleatoria de obstáculos (la "suerte" de la semilla afecta mucho el resultado), conviene **1.000+ partidas** para que el ruido de la semilla se promedie y el modelo capture patrones de habilidad reales y no la suerte de una tirada particular.

**Riesgo si el dataset está mal definido:**
Si por error se define la fila como "un frame" y se usa `score_actual` (el puntaje corriente en ese frame) como si fuera el objetivo, el modelo tendría acceso trivial a la respuesta (el score en el frame 900 predice casi perfectamente el score final si la partida dura 1.000 frames): esto es una fuga de información (data leakage). El modelo daría métricas espectaculares en entrenamiento pero sería inútil para el objetivo real, que es predecir el resultado **antes** de que la partida transcurra, con información temprana (config, estilo de juego, no el score acumulado).

---

### P3 — ¿Qué tipo de obstáculo viene próximo?

**Variable objetivo (Y):** `tipo_obstaculo_siguiente` — **categórica multiclase** (p. ej. cactus_1, cactus_2, cactus_3, pajaro_bajo, pajaro_alto, ninguno_visible_aun).

**Variables de entrada (X)** — mínimo 5:

| Variable | Por qué |
|---|---|
| `tipo_obstaculo_anterior` (t-1) | Si el generador tiene algun patron en los obstáculos. |
| `tipo_obstaculo_anterior_2` (t-2) | Permite detectar patrones de orden 2 no solo de orden 1. |
| `racha_del_mismo_tipo` (cuántos seguidos iguales) | Muchos generadores de juegos evitan 3+ repeticiones idénticas; esta variable prueba esa hipótesis. |
| `velocidad_actual` / `dificultad_actual` | Es común que el generador ligue la dificultad (más pájaros, menos espacio) al tiempo transcurrido. |
| `distancia_desde_ultimo_obstaculo` | Puede reflejar reglas de espaciado mínimo del generador, correlacionadas con el tipo que sigue. |

**Granularidad:** un evento = la aparición de un nuevo obstáculo (no cada frame). La mayoría de los frames entre dos obstáculos no aportan información nueva sobre "cuál viene después"; usar granularidad de frame aquí solo infla el dataset con filas redundantes.

**Advertencia de diseño clave:** antes de construir cualquier clasificador para P3 hay que verificar, con estadística simple, si la secuencia de obstáculos es realmente generada por un proceso con estructura o si es esencialmente aleatoria. Si resulta ser puramente aleatoria, **el modelo correcto no es un clasificador, habría que investigar cual nos puede servir** 

**Tamaño mínimo razonable:** con ~6 tipos de obstáculo, una tabla de transición tiene 6×6 = 36 pares posibles. Para tener confianza estadística mínima (≈30–50 observaciones por par, regla práctica para frecuencias en tablas de contingencia) se necesitan **al menos 1.000–2.000 eventos de transición**, lo que en partidas típicas (con ~15–25 obstáculos cada una) equivale a aproximadamente **80–150 partidas** solo para el análisis de patrón; para entrenar un modelo conviene varias veces esa cifra.

**Riesgo si el dataset está mal definido:**
Dos riesgos simétricos:
1. Asumir que hay patrón cuando no lo hay → el modelo memoriza ruido de la muestra de entrenamiento (overfitting a la semilla usada) y falla en producción.
2. Descartar el problema como "puro azar" sin muestra suficiente → se pierde una regla real del generador (p. ej. "no más de 2 pájaros seguidos"), y se tira a la basura una señal útil.

---

## Resumen comparativo

| | P1 (frame → morirá) | P2 (partida → score) | P3 (evento → tipo de obstáculo) |
|---|---|---|---|
| Tipo de Y | Binaria | Numérica continua | Categórica multiclase |
| Granularidad de fila | Frame (~16 ms) | Partida completa | Evento (aparición de obstáculo) |
| Riesgo principal | Ventana de etiquetado mal definida → modelo no anticipa | Fuga de información (score actual filtra el score final) | Sobre-modelar ruido si la secuencia es aleatoria |
| Tamaño mínimo estimado | ~500–5.000 partidas | ~100–1.000+ partidas | ~80–150 partidas (para probar patrón) |
| Desafío central de diseño | Desbalance extremo de clases | Definir bien qué es "información temprana" vs. "información filtrada" | Confirmar que existe señal antes de modelar |

### Misión 2: Diccionario de datos (qué debe traer el CSV)

Revisé la muestra de 10 filas (sesión 7 = frames 0, 40, 80, 81, 82; el resto son filas de otras sesiones). Dos observaciones antes de responder: la muestra **no trae la columna `session_id`** aunque el diccionario sí la lista (las sesiones solo se distinguen porque el `frame` se reinicia en 0), y la décima fila (frame 50) no está cubierta por la nota del enunciado, así que la trato como otra sesión.

**1. ¿Qué patrón se ve en la fila con `died=1` (frame 82)?**

| frame | time_ms | speed | obstacle_type | dist_obstacle | jump | died |
|---|---|---|---|---|---|---|
| 80 | 1280 | 6.8 | cactus_small | 55 | 1 | 0 |
| 81 | 1296 | 6.8 | cactus_small | 38 | 1 | 0 |
| **82** | 1312 | 6.8 | cactus_small | **12** | **0** | **1** |

- La distancia al obstáculo se acorta rápido (55 → 38 → 12) y en el frame de la muerte es la más baja de toda la sesión.
- El dino venía saltando en los frames 80 y 81 y en el 82 ya aparece con `jump=0`: choca justo cuando deja de estar en el aire (saltó demasiado pronto o aterrizó encima del cactus).
- La otra muerte de la muestra (frame 280, `cactus_large`, `dist=22`) también tiene `jump=0` y distancia pequeña. Con solo dos casos es una pista, no una regla: ambos choques ocurren con `dist < 25` y `jump=0`.
- Detalle a revisar: entre frames consecutivos la distancia baja 17 y luego 26 px, aunque `speed` se queda en 6.8. Eso sugiere que `speed` y `dist_obstacle` no están en las mismas unidades, y el diccionario no dice cuáles son.

**2. ¿`score` sirve para predecir la muerte en el siguiente frame?**

No. En la muestra `score = time_ms / 80` en todas las filas (640/8, 1280/16, 1920/24, 3200/40, 4480/56, 800/10), y `speed = 6.0 + time_ms/1600` también se cumple en todas. Es decir, `score`, `speed` y `time_ms` son la misma señal: cuánto lleva jugando la partida. Eso dice qué tan difícil va el juego, pero no si el obstáculo está a punto de alcanzar al dino (en los frames 80 a 82 el `score` ni cambia, se queda en 16). Además es una variable redundante (colinealidad perfecta) y, usada sola, aprendería el atajo "partida larga → más riesgo" en vez de la causa real.

**3. ¿Falta alguna columna crítica para P1?**

Sí. Las que más pesan:
- **Altura del dino o velocidad vertical:** `jump` solo dice si está en el aire, no si va subiendo o bajando ni si alcanzará a librar el obstáculo.
- **Estado agachado:** sin esto no se puede evaluar los pájaros bajos (se esquivan agachándose, no saltando).
- **Tamaño del obstáculo** (ancho y alto) y la **altura del pájaro**: `obstacle_type` es una etiqueta, no una medida de la hitbox.
- **Lag de reacción del jugador** (tiempo entre que aparece el obstáculo y la tecla).
- **`session_id` presente en cada fila**, para poder dividir train y test por partida.
- Se puede derivar `tiempo_hasta_choque = dist_obstacle / speed`, pero antes hay que confirmar las unidades.

**4. ¿`died` tal como está definida sirve para P1?**

No directamente. Vale 1 solo en el último frame de la sesión, o sea en el frame del choque. Para P1 la etiqueta correcta es la del frame anterior: `Y(t) = died(t+1)` dentro de la misma sesión (el último frame no tiene siguiente y se descarta). Así como está, `died` describe cómo terminó la partida y entrenar con ella enseñaría a reconocer un choque que ya pasó. Además deja un solo 1 por partida, lo que produce el desbalance extremo de la Misión 4.

---

## Bloque 2 — Preguntas que todo EDA debe responder

### Misión 3: Las diez preguntas del analista

Se eligen tres preguntas del checklist: **#3 (balance de clases)**, **#4 (leakage)** y **#8 (i.i.d.)**, por ser las más determinantes para P1 (¿morirá en el siguiente frame?).

**#3 — ¿La clase objetivo está balanceada?**
R: si una partida promedio dura ~240 frames (a 60 fps, ~4 s) y cada partida termina en exactamente un frame de muerte, entonces el dataset tiene tantos "unos" como partidas. Con 50 partidas y ~12.000 frames totales, eso da **50 frames positivos sobre 12.000** → **≈0,42% de la clase positiva** (razón aproximada 1:239). Es un desbalance extremo, del mismo orden que se estimó en el Bloque 1.

**#4 — ¿Hay fugas de información (leakage)?**
R: si la tubería de datos calcula una columna como `tiempo_restante_hasta_morir = tiempo_de_muerte − tiempo_actual` (es decir, mira hacia el futuro para construirla), esa columna es casi un predictor perfecto del frame de muerte — pero es inútil en producción, porque en el momento de predecir en vivo no se conoce el instante futuro de la colisión. El mismo riesgo aparece con `score`: como el score crece con el tiempo de partida, un valor de score "típico de partidas que ya casi terminan" puede filtrar información sobre la proximidad de la muerte sin que el modelo entienda realmente la causa (obstáculo cercano), sino que aprenda un atajo estadístico ("score alto → probablemente cerca de morir") que no generaliza a partidas con dificultad distinta.

**#8 — ¿Los datos son i.i.d.? (independientes e idénticamente distribuidos)**
R: mezclar frames de la misma partida entre entrenamiento y prueba es un error porque los frames consecutivos están fuertemente autocorrelacionados: la posición, velocidad y distancia al obstáculo cambian muy poco de un frame al siguiente (a 60 fps, dos frames seguidos son casi el mismo estado). Si se hace una partición aleatoria por fila, frames casi idénticos de una misma partida terminan repartidos entre train y test, y el modelo "reconoce" ese contexto en vez de generalizar a partidas nuevas — esto infla artificialmente las métricas de validación (una forma de leakage por duplicados casi exactos). La partición correcta es **por partida completa** (todos los frames de una partida van juntos a train o a test, nunca repartidos), por ejemplo con un group k-fold usando el id de partida como grupo.

---

### Misión 4: Leer distribuciones sin gráfica (interpretación)

**1. ¿P1 está desbalanceado? Cuantifica.**

Sí, muy. Hay 50 unos en ~12 000 frames: 50 / 12 000 ≈ **0,42 %** de la clase positiva. Quedan unos 11 950 ceros contra 50 unos, una razón de aproximadamente **1 : 239**. Un modelo que siempre responda "no muere" acertaría el 99,58 % de los frames y no detectaría ni una muerte.

**2. ¿Qué implica para la métrica?**

`accuracy` no sirve: premia al modelo que nunca avisa. Hay que mirar la clase positiva: **recall** (de las muertes reales, cuántas anticipa), **precision** (de las alarmas, cuántas eran reales) y **F1** como resumen, junto con la matriz de confusión. Además hay que ponderar las clases (`class_weight`) o ajustar el umbral de decisión, y evaluar por partida y no por fila suelta.

**3. ¿`dist_obstacle` parece útil como predictor?**

Sí, pero no alcanza por sí sola. Hay señal: el analista previo anota que las muertes suelen darse con `dist < 20`, y las dos muertes de la muestra tienen 12 y 22 px, contra una media de 95 y una mediana de 88 en todo el dataset. Es decir, en el momento de morir la distancia está muy por debajo de lo normal. Sin embargo: (a) una de las dos muertes de la muestra (22) queda justo fuera del umbral de 20, (b) el mínimo global es 5, así que hay frames con distancias muy chicas en los que el dino no muere (los salvó el salto), y (c) la distancia peligrosa depende de la velocidad (`speed` llega hasta 13) y de si el dino está en el aire. Por eso `dist_obstacle` debe combinarse con `jump` y `speed`.

**4. ¿La distribución de `score` sugiere regresión simple o transformación?**

La media (28) es bastante mayor que la mediana (18) y el máximo (120) está muy lejos: es una **cola larga a la derecha**. Una regresión lineal simple sobre el `score` crudo se vería afectada por esa cola, así que conviene transformarlo (por ejemplo `log(1 + score)`) o usar un modelo robusto, como un árbol regresor. Dos advertencias: este resumen es del `score` **por frame**, y para P2 la unidad es la partida completa, de modo que solo habría 50 filas; con tan pocas partidas hay que mantener el modelo muy simple y no sobreinterpretar.

---

## Bloque 3 — Del EDA a la elección del modelo

### Misión 5: Árbol de decisión "dataset → modelo"

**Cómo leo la tabla guía.** P1 cae en la fila *Y binaria muy desbalanceada* (y también en *secuencia de frames por sesión*). P2 cae en *Y numérica*. P3 cae en *Y categórica multiclase*.

| Escenario | Tras mi EDA, ¿qué fila de la guía aplica? | Modelo que propondría | 2 condiciones del dataset que deben cumplirse |
|---|---|---|---|
| **P1** | *Y binaria muy desbalanceada* (con estructura de secuencia por sesión) | Árbol de decisión poco profundo o regresión logística con `class_weight='balanced'` y umbral ajustado; Random Forest si hay suficientes datos. Se evalúa con recall, precision y F1 de la clase "muere". | (1) Suficientes positivos reales (del orden de 500 o más) con la etiqueta construida como `died(t+1)` y sin información del futuro. (2) La partición train/test se hace **por partida** (group k-fold con `session_id`), no por fila. |
| **P2** | *Y numérica (score final)* | Regresión lineal como línea base y árbol regresor (o Random Forest regresor) si hay relaciones no lineales. | (1) Una fila por partida con variables agregadas disponibles **antes** de terminar, no el score acumulado. (2) Suficientes partidas (mínimo 100 a 200, idealmente más de 1 000) y una `Y` sin colas extremas, o transformada con log. |
| **P3** | *Y categórica multiclase* | Árbol de decisión o regresión logística multinomial, comparados contra una línea base (tabla de transiciones o la clase más frecuente). | (1) Evidencia estadística de que la secuencia de obstáculos **no es puramente aleatoria** (por ejemplo, una prueba chi-cuadrada sobre las transiciones). Si es aleatoria, no se modela. (2) Suficientes eventos por par de tipos (unos 30 a 50 por celda, o sea 1 000 a 2 000 transiciones). |

**Modelos poco razonables (y por qué):** una red profunda para P1 sin muchos más datos (se sobreajusta con 50 positivos); usar accuracy como única métrica en P1; un clasificador binario para P2 (la `Y` es un número); una regresión lineal sobre códigos 1, 2, 3 para P3 (inventa un orden que no existe entre tipos de obstáculo); ignorar el orden temporal de los frames.

---

### Misión 6: Contraejemplo — cuándo no usar un modelo

**1. Un árbol profundo parece buena idea pero el EDA lo desaconseja.**
En P1, con 50 muertes en 12 000 frames, un árbol sin límite de profundidad puede crear una hoja para casi cada muerte y sacar métricas casi perfectas en entrenamiento. El EDA lo desaconseja por tres motivos que ya vimos: hay muy pocos positivos (50), los frames de una misma partida son casi idénticos (no son i.i.d.) y, si se parte el dataset por fila, esos frames casi duplicados caen en train y en test y el modelo solo "memoriza partidas". Resultado: validación inflada y desempeño pobre con partidas nuevas.

**2. Un escenario donde una red neuronal sí tendría sentido.**
Predecir P1 con una **ventana de los últimos N frames** (distancia, velocidad, altura, tipo de obstáculo) cuando el EDA muestra: (a) muchas partidas, del orden de miles, con cientos o miles de positivos; (b) relaciones no lineales entre variables (el peligro depende de la combinación distancia × velocidad × fase del salto) que un árbol corto o una regresión logística no capturan, y que las líneas base quedaron por debajo; y (c) dependencia temporal clara entre frames (autocorrelación). Con eso se justifica una red MLP sobre la ventana o una GRU/LSTM sobre la secuencia.

**3. ¿Se podría resolver P1 con reglas fijas? Comparación.**
Sí, una primera versión sería: *si `dist_obstacle < X` y `jump = 0`, entonces muerte inminente*. En la muestra funcionaría con X ≈ 25 (atrapa las muertes con 12 y 22 px).

| | Regla fija | Modelo aprendido |
|---|---|---|
| Ventajas | No necesita datos de entrenamiento, es transparente, se audita en segundos y casi no tiene costo de cómputo. Sirve como línea base que el modelo debe superar. | Combina `dist`, `speed`, `obstacle_type`, `jump` y altura, y entrega una probabilidad en vez de un sí/no. |
| Límites | El umbral X depende de la velocidad (`speed` va de 6 a 13) y del tipo de obstáculo; con un X fijo se generan falsas alarmas a baja velocidad y se llega tarde a alta velocidad. No distingue pájaros (se esquivan agachándose). | Necesita suficientes positivos y una partición por partida; puede sobreajustarse y es menos fácil de explicar. |

Una opción razonable es una regla basada en `dist_obstacle / speed` (tiempo hasta el choque) como línea base, y entrenar el modelo solo si logra mejorarla de forma clara.

---

## Síntesis (5 líneas)

1. Antes de pedir un modelo hay que fijar qué es una fila y qué es `Y`: P1 es por frame con `Y = died(t+1)`, P2 por partida y P3 por evento de obstáculo.
2. El primer dataset que pediría es la telemetría por frame con `session_id`, altura y velocidad vertical del dino, estado agachado y tamaño del obstáculo, de al menos 500 partidas.
3. El EDA ya muestra tres riesgos: desbalance de 1 : 239 en P1, variables redundantes (`score`, `speed` y `time_ms` son la misma señal) y frames no independientes dentro de una partida.
4. Con eso, el modelo vendría **solo después**: árbol poco profundo o regresión logística con clases ponderadas para P1, evaluado con recall y F1 y partido por partida.
5. Una regla fija sobre `dist_obstacle / speed` sirve de línea base: un modelo más complejo solo se justifica si la supera con claridad.
