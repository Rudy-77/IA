
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
