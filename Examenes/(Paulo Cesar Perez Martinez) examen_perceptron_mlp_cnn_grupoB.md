# EXAMEN TEÓRICO — PERCEPTRÓN, MLP Y CNN (GRUPO B)

**Eduardo Alcaraz**

## Datos del alumno

| Campo | Valor |
|---|---|
| Nombre | Paulo Cesar Perez Martinez |
| Grupo | B |
| Fecha | 3/Nov/2026 |

---

## Pregunta 1 — Por qué no basta “una sola neurona”

1. Resume el funcionamiento de un perceptrón como “regla de decisión”.
2. Explica la noción de **frontera lineal** sin escribir fórmulas.
3. Da un ejemplo (aunque sea inventado) donde esa frontera **no** alcanza.

### Respuesta

**1.**

Un perceptrón es, en esencia, un pequeño votante con criterio propio. Recibe varias señales de entrada, le da a cada una una importancia y suma todo. Si el resultado total supera cierto umbral, responde “sí”; si no lo alcanza, responde “no”.

Entrenarlo consiste en ajustar esas importancias y ese umbral hasta que sus respuestas coincidan con los ejemplos que le mostraste. Al final es una regla fija del tipo: “si la combinación ponderada de las entradas es suficientemente grande, haz tal cosa si no, no”.

**2.**

Imaginemos cada ejemplo como un punto en un plano: un eje es la velocidad de la bala y el otro es la distancia. Los puntos donde hay que saltar son de un color y los de no saltar, de otro.

Lo que hace el perceptrón es trazar una sola línea recta sobre ese plano, de un lado de la línea dice “salta” y del otro “no saltes”. Esa línea es la frontera lineal puede inclinarse y desplazarse, pero siempre será recta, no puede doblarse, curvarse ni hacer varios tramos.

**3.**

Suponiendo una variante inventada de un juego con esta regla: el jugador debe saltar solo cuando la bala está a una distancia “intermedia”. Si está muy lejos, todavía no hace falta saltar; si está muy cerca, ya es tarde.

En el plano, los puntos de “salta” forman una franja en medio, con puntos de “no saltes” a ambos lados. Ninguna línea recta puede separar una franja central de sus dos lados: cualquier recta que dejes bien de un lado, deja mal clasificado el otro.

El ejemplo clásico es el XOR (“uno u otro, pero no ambos”): imagina que debes saltar cuando la bala es rápida o cercana, pero no cuando es ambas cosas a la vez ni ninguna. Los puntos de “salta” quedan en esquinas opuestas del plano, y los de “no saltes” en las otras dos esquinas. Cualquier recta que se traze deja siempre al menos un punto del lado equivocado.

Cuando la relación entre las entradas y la decisión tiene “curvas”, “franjas” o “zonas separadas”, una sola neurona se queda corta. Por eso se apilan neuronas en capas, cada neurona traza una recta y la capa siguiente las combina, logrando fronteras curvas o de varios tramos.

---

## Pregunta 2 — CNN: de píxeles a mapas

1. ¿Qué problema de **estructura espacial** ignora un MLP si aplana la
   foto?
2. ¿Qué significa que un mismo filtro se aplique en toda la imagen?
3. Además de clasificación de objetos, cita **dos** usos donde la CNN
   encaja bien.

### Respuesta

**1.**

Al aplanar una imagen, la conviertes en una fila larga de números y se pierde el “quién está junto a quién”. Para un MLP, el píxel de la esquina superior izquierda y el que está justo a su lado no tienen ninguna relación especial son entradas sueltas, como si alguien hubiera barajado las columnas de una tabla.

Eso trae dos problemas concretos:

- **Se pierde la vecindad.** Bordes, texturas y formas dependen de que píxeles cercanos se vean juntos, y el MLP no sabe cuáles lo son.
- **No generaliza por posición.** Si un gato aparece arriba a la izquierda en una foto y abajo a la derecha en otra, el MLP lo ve como patrones completamente distintos y tendría que aprender cada caso por separado.

Además, necesita un peso por cada píxel hacia cada neurona, lo que dispara la cantidad de parámetros.

**2.**

Un filtro es una ventanita pequeña que busca un patrón concreto como un borde vertical, una esquina, una mancha de cierto color. Que se aplique en toda la imagen significa que esa misma ventana, con los mismos valores, se desliza por todas las posiciones y pregunta en cada una “¿hay aquí lo que yo busco?”.

Dos consecuencias importantes:

- **Reutilización de pesos:** en vez de aprender un detector distinto para cada zona, se aprende uno solo y se usa en todas esto reduce muchísimo los parámetros.
- **Detección sin importar la posición:** si el patrón aparece en otro lugar, el mismo filtro lo encuentra lo que sale es un mapa de activación que indica dónde se encontró el patrón y con qué intensidad.

Al apilar capas, los primeros filtros detectan cosas simples y los siguientes combinan esos mapas para reconocer formas más complejas.

**3.**

- Detección y segmentación de imágenes médicas por ejemplo; localizar tumores o lesiones en radiografías, resonancias o tomografías, marcando qué píxeles pertenecen a la zona afectada.
- Reconocimiento y lectura de texto (OCR) o de señales de tráfico en vehículos autónomos identificar caracteres, placas o señales y ubicar peatones y carriles en el video de la carretera.

---

## Pregunta 3 — Softmax, clases y decisiones

1. Diferencia un problema **binario** de uno **multiclase** en términos de
   salida de la red.
2. ¿Qué lectura le das a las probabilidades que produce un softmax?
3. Imagina tres carpetas de imágenes (tres riesgos o tres animales):
   ¿qué arquitectura de **salida** usarías y por qué?

### Respuesta

**1.**

**Binario** : la red necesita una sola neurona de salida. Su valor se interpreta como “qué tan probable es la clase positiva”. Si pasa de cierto umbral, dice “sí”; si no, dice “no”. La probabilidad de la otra clase es simplemente lo que le falta a uno.

**Multiclase** : la red necesita una neurona de salida por clase. Cada neurona produce un puntaje para su clase, y la decisión consiste en comparar todos los puntajes y quedarse con el más alto. Ya no basta un solo número, porque la pregunta cambió de “¿sí o no?” a “¿cuál de estas opciones?”.

**2.**

Softmax toma esos puntajes crudos y los convierte en números entre 0 y 1 que suman exactamente 1. La lectura natural es: “de todo mi grado de certeza, este porcentaje se lo doy a cada clase”.

Es un reparto de confianza entre opciones que se excluyen mutuamente. Si una clase sube, las demás bajan. Por eso solo sirve cuando cada ejemplo pertenece a una sola clase.

La clase ganadora es la de mayor probabilidad, pero el valor también dice qué tan segura está la red. Un 0.95 es una decisión firme; un 0.40 contra 0.35 y 0.25 es una duda real, y quizá convenga no actuar automáticamente.

No es una probabilidad garantizada del mundo real. Una red mal calibrada o enfrentada a algo que nunca vio (un perro cuando solo conoce gato, caballo y vaca) igual repartirá su 100% entre las clases que conoce, a veces con mucha seguridad y sin razón.

**3.**

Usaría una capa de salida con 3 neuronas y activación softmax, entrenada con entropía cruzada categórica.

**Por qué:**

- **Tres neuronas**, una por carpeta, porque el problema es multiclase y cada neurona representa una categoría, por ejemplo; riesgo bajo, medio y alto, o tres animales.
- **Softmax**, porque cada imagen pertenece a una sola carpeta; las clases se excluyen entre sí y queremos un reparto de probabilidad que sume 1 y permita elegir la más probable.
- **Entropía cruzada**, porque es la función de pérdida que encaja con softmax: castiga con fuerza que la red le dé poca probabilidad a la clase correcta.

---

## Pregunta 4 — MLP: profundidad con sentido

1. ¿Qué ganas al apilar capas densas con activaciones no lineales?
2. ¿Qué riesgo aparece si la red es **demasiado** grande para tus datos?
3. Relaciona “capa oculta” con la idea de aprender representaciones
   intermedias (en palabras simples).

### Respuesta

**1.**

Ganas la capacidad de aprender fronteras que no son rectas. Una sola neurona traza una línea; una capa de varias neuronas traza varias líneas, y la capa siguiente las combina para formar curvas, franjas, islas o zonas separadas. Con eso se resuelven casos como el XOR o el de la “franja intermedia”.

El detalle importante está en la palabra no lineales. Si se apilan capas sin activación no lineal, la combinación de varias capas equivale matemáticamente a una sola capa: seguiría con una frontera recta, solo que con más pasos para llegar a ella. La no linealidad es lo que hace que cada capa extra aporte algo nuevo.

**2.**

El riesgo principal es el sobreajuste (overfitting): la red tiene tanta capacidad que, en lugar de aprender la regla general, memoriza los ejemplos de entrenamiento, incluido su ruido y sus casualidades. Se nota porque el error en entrenamiento baja mucho, pero el error en datos nuevos (validación o prueba) se queda alto o empeora.

Esto pesa más cuando los datos son pocos. Con alrededor de mil filas como las de datos_mlp.csv, una red con muchas neuronas y capas tiene más parámetros que información que aprender. Además, cuesta más entrenarla y es más difícil de interpretar.

Formas comunes de controlarlo:

- Reducir el tamaño de la red con menos capas o neuronas.
- Separar datos de validación y vigilar cuándo deja de mejorar.
- Regularización.
- Conseguir más datos o aumentarlos.

**3.**

Una capa oculta es un paso intermedio de “traducción”: no recibe los datos originales ni da la respuesta final, sino que transforma la entrada en algo más útil para decidir.

En palabras simples la red no mira directamente “velocidad = 7, distancia = 120” para decidir. Primero la capa oculta aprende a calcular cosas como “¿la bala está en zona de peligro?” o “¿llegará pronto?”, que son conceptos nuevos construidos a partir de las entradas. Nadie le dice cuáles conceptos formar; los descubre durante el entrenamiento porque le sirven para acertar. La capa de salida después solo tiene que combinar esos conceptos para dar la decisión.

Es la misma idea de la CNN, los primeros filtros detectan bordes, los siguientes combinan bordes en formas y los últimos reconocen objetos. Cada capa trabaja con el resultado de la anterior y lo vuelve un poco más abstracto, hasta que la decisión final resulta sencilla

---

## Pregunta 5 — Entrenamiento sin magia

1. Explica qué es una **época** a nivel intuitivo.
2. ¿Hacia qué “objetivo” se mueven los pesos cuando hablamos de
   minimizar el error?
3. Si el learning rate es absurdo (muy grande o muy chico), ¿qué
   síntomas conceptuales esperarías?

### Respuesta

**1.**

Una época es una vuelta completa por todos los datos de entrenamiento. Es como estudiar para un examen repasando todo el cuaderno de principio a fin y cuando terminas de ver todos los ejemplos una vez, pasó una época.

Entrenar casi nunca se logra en una sola vuelta, así que se repite muchas épocas. En cada una, la red ve los mismos ejemplos normalmente en otro orden y por lotes pequeños y ajusta un poco sus pesos.

**2.**

Se mueven hacia donde el error es más pequeño. Ese error se mide con una función de pérdida, un número que dice qué tan equivocadas están las predicciones frente a las respuestas reales.

En una imagen como un paisaje de montañas y valles, donde la altura en cada punto es el error que tendría la red con cierta combinación de pesos. Entrenar es bajar por la pendiente hasta llegar a un valle, es decir, a una combinación de pesos con error bajo. En cada paso, el algoritmo mira hacia dónde baja más el terreno y da un paso en esa dirección.

Conviene recordar que el objetivo real no es el valle de los datos de entrenamiento, sino que la red funcione bien con datos nuevos.

**3.**

El learning rate es el tamaño del paso que das al bajar la pendiente.

**Demasiado grande:**

- El error oscila o se dispara en lugar de bajar de forma estable.
- Es como dar zancadas tan largas que pasas por encima del valle y caes al otro lado, una y otra vez.
- En casos extremos, el error crece sin control o aparecen valores inválidos o NaN.
- La red nunca se estabiliza, aunque pasen muchas épocas.

**Demasiado pequeño:**

- El error baja, pero tan lentamente que parece estancado.
- Son pasos de hormiga: necesitarías muchísimas épocas para llegar a algún lado.
- Puede quedarse atascado en una zona plana o en un valle mediocre y dar la falsa impresión de que ya “aprendió todo”.
- El entrenamiento resulta caro en tiempo sin mejora real.

**Señal práctica:** graficar el error por época. Una curva que baja suave y se aplana indica un buen ritmo; una curva en zigzag o que sube indica paso demasiado grande; una casi horizontal desde el inicio indica que es demasiado pequeño.

---

## Pregunta 6 — Bloques de una CNN

Sin matemáticas, di el papel de cada pieza:

1. Kernel / filtro convolucional
2. Función de activación después de conv
3. Max pooling
4. Capa densa + softmax al final de un clasificador

### Respuesta

**1.**

Es el detector de patrones. Una ventanita pequeña que se desliza por la imagen preguntando en cada posición: “¿aparece aquí el patrón que yo busco?”.

Cada filtro busca algo distinto y produce un mapa que indica dónde y con qué fuerza lo encontró. Sus valores no se diseñan a mano, la red los aprende durante el entrenamiento.

**2.**

Aporta la no linealidad, igual que en el MLP. Sin ella, apilar capas convolucionales equivaldría a una sola operación simple. Con ReLU, por ejemplo, lo que queda es: “me quedo con las respuestas positivas y descarto las negativas”. Así el mapa resalta solo lo relevante y la red puede combinar patrones simples en otros más complejos.

**3.**

Es el resumidor. Divide el mapa en bloques pequeños y se queda solo con el valor más fuerte de cada bloque. Esto logra tres cosas:

- **Reduce el tamaño del mapa**, así hay menos cálculos y menos parámetros.
- **Da tolerancia a pequeños desplazamientos**: si el patrón se movió un poco dentro del bloque, el resultado casi no cambia.
- **Conserva lo esencial**: lo que importa es que el patrón apareció, no su posición exacta al píxel.

**4.**

Las capas anteriores se encargaron de extraer características (qué patrones hay en la imagen). Aquí empieza la decisión:

- **Capa densa:** recibe todas esas características ya aplanadas y las combina, como un MLP, para decidir qué conjunto de rasgos corresponde a qué clase.
- **Softmax:** convierte los puntajes finales en un reparto de probabilidad entre las clases (suman 1), de modo que se pueda elegir la más probable y ver qué tan segura está la red.

> El filtro detecta, la activación filtra lo útil, el pooling resume y la capa densa con softmax decide.

---

## Pregunta 7 — XOR y la historia del aprendizaje

1. ¿Por qué XOR se volvió un “contraejemplo” famoso?
2. ¿Qué elemento de un MLP permite patrones que no son una sola línea?
3. ¿En qué tipo de tabla/problema del mundo real seguirías usando algo
   tan simple como un perceptrón o un modelo lineal?

### Respuesta

**1.**

XOR es el caso más pequeño y claro de un problema que ninguna línea recta puede separar. Con dos entradas binarias, las combinaciones que dan 1 "0,1 y 1,0" quedan en esquinas opuestas del plano, y las que dan 0 "0,0 y 1,1" en las otras dos. Cualquier recta deja al menos un punto del lado equivocado.

Se hizo famoso en los años 60, cuando Minsky y Papert lo usaron en su libro Perceptrons para señalar los límites de las redes de una sola neurona. Aunque su crítica era sobre perceptrones de una capa, mucha gente la interpretó como que las redes neuronales en general no llevaban a nada, y el interés y el financiamiento cayeron durante años. El tema se retomó en los años 80, cuando la retropropagación hizo práctico entrenar redes con varias capas, que sí resuelven XOR.

**2.**

Son las capas ocultas con activaciones no lineales. Cada neurona oculta traza su propia recta, y la capa siguiente las combina. Para XOR bastan dos neuronas ocultas: una detecta “al menos una entrada activa” y otra “ambas activas”, y la salida combina las dos “lo primero sí, lo segundo no”. Sin la no linealidad, las capas se colapsarían en una sola recta.

**3.**

En datos tabulares donde la relación es aproximadamente lineal o los datos son pocos, por ejemplo:

- **Scoring de riesgo crediticio o clínico** con variables como edad, ingreso o presión arterial, donde un modelo lineal es suficiente y se puede explicar a un auditor o a un médico: cada variable tiene un peso que se puede leer.
- **Predicción de ventas o demanda** a partir de pocas variables "precio, temporada, publicidad", como línea base rápida.
- **Clasificación de texto** con muchas características como filtros de spam con bolsa de palabras, donde los modelos lineales funcionan muy bien porque los datos de alta dimensión suelen ser casi separables.

**Razones para elegirlos:** son rápidos, necesitan pocos datos, se interpretan fácilmente y, si el problema no tiene curvas, una red grande solo añadiría riesgo de sobreajuste. Una buena práctica es empezar con un modelo lineal como referencia y complicar solo si realmente mejora.

---

## Pregunta 8 — De lo simple a lo abstracto en visión

1. Contrasta lo que buscan capas tempranas vs capas profundas en una
   CNN.
2. ¿Qué intuición da eso para **transfer learning** (aunque no lo hayas
   programado)?
3. ¿Cuándo sospecharías que tu CNN es “profunda de más” para el
   dataset del curso?

### Respuesta

**1.**

- **Capas tempranas:** buscan patrones simples y locales: bordes, líneas en distintas orientaciones, esquinas, manchas de color, texturas muy básicas. Cada filtro mira una zona pequeña de la imagen.
- **Capas profundas:** combinan los mapas de las capas anteriores para detectar patrones complejos y abstractos: partes de objetos y, más adelante, objetos completos o conceptos. Cada neurona “ve” una zona más amplia de la imagen porque se apoya en todo lo construido antes.

Cada capa crea representaciones intermedias más abstractas que la anterior. Un borde se vuelve esquina, la esquina se vuelve forma, la forma se vuelve parte y la parte se vuelve objeto.

**2.**

Los bordes, texturas y formas básicas son casi universales: sirven igual para reconocer gatos, radiografías o señales de tránsito. Lo específico de cada tarea está sobre todo en las capas profundas y en la capa final.

Por eso se puede tomar una CNN ya entrenada con millones de imágenes (por ejemplo, ImageNet), conservar sus capas tempranas (que ya saben “ver”) y reentrenar solo las últimas capas con tu dataset pequeño. Ventajas:

- Necesitas muchos menos datos.
- Entrenas más rápido.
- Reduces el riesgo de sobreajuste, porque no aprendes todo desde cero.

Con datasets muy parecidos al original basta cambiar la capa final; con datasets distintos conviene reentrenar también algunas capas intermedias o profundas.

**3.**

Las señales típicas, que conectan con el sobreajuste de la pregunta 4:

- **Brecha grande entre entrenamiento y validación:** la exactitud en entrenamiento llega casi al 100% pero en validación se queda mucho más baja o empeora con las épocas.
- **Entrenamiento inestable o muy lento sin ganar nada de precisión frente a una red más pequeña.**
- **Una red más simple rinde igual o mejor:** si con menos capas o filtros obtienes resultados parecidos, las capas extra solo añaden parámetros sin aportar.
- **Más parámetros que ejemplos:** con un dataset pequeño, una red con millones de parámetros tiene capacidad de sobra para memorizar.
- **Fallas con pequeñas variaciones:** acierta en imágenes de entrenamiento pero se equivoca al rotar, recortar o cambiar un poco la iluminación.

**Qué hacer:** reducir capas o filtros, aplicar aumento de datos y regularización o, mejor aún, usar transfer learning en lugar de entrenar una red profunda desde cero.

---

## Pregunta 9 — Análisis: matching problema → modelo

Elige **Perceptrón/lineal** / **MLP** / **CNN** (una opción) y justifica en
una frase:

1. Predecir “compra / no compra” con edad, ingreso y visitas a la web
   (columnas numéricas).
2. Detectar si hay un gato en una foto.
3. Aproximar una frontera “curva” o a trozos en datos 2D tabulares.
4. ¿CNN es la herramienta natural para predecir la siguiente palabra
   de un párrafo? Justifica y nombra una familia del curso más
   adecuada (RNN/Transformer a nivel nombre).

### Respuesta

**1.**

**Perceptrón/lineal.** Son pocas columnas numéricas y la relación suele ser aproximadamente lineal, así que un modelo lineal basta, es rápido y se puede explicar. Si la línea base se queda corta, el siguiente paso sería un MLP.

**2.**

**CNN.** Sus filtros aprovechan la vecindad de los píxeles y detectan bordes, formas y partes del gato sin importar en qué zona de la imagen aparezca (preguntas 2 y 8).

**3.**

**MLP.** Sus capas ocultas con activaciones no lineales combinan varias rectas para formar curvas y zonas separadas, y los datos tabulares no tienen estructura espacial que justifique una CNN.

**4.**

**No es la más natural.** Una CNN está pensada para estructura espacial en rejilla, y en el lenguaje lo que importa es el orden y el contexto a lo largo de una secuencia, con dependencias que pueden ser lejanas. La familia más adecuada es Transformer, o las RNN (LSTM/GRU) como enfoque anterior que procesa palabra por palabra con memoria.

---

## Pregunta 10 — Análisis: auditoría de un equipo

Un equipo entrega un modelo “muy preciso” en su laptop y falla con
datos nuevos.

1. ¿Qué diagnósticos conceptuales harías **antes** de tocar
   hiperparámetros al azar?
2. Pide **dos** evidencias que deberían mostrar en la defensa (curvas,
   separación train/val, ejemplos de error, etc.).
3. Redacta **tres** preguntas tuyas para comprobar si saben cuándo usar
   perceptrón, MLP o CNN (distintas a las del Grupo A si las conoces).

### Respuesta

**1. Diagnósticos conceptuales antes de tocar hiperparámetros**

Primero hay que entender por qué falla, no probar valores al azar:

- **¿Es sobreajuste?** Comparar el desempeño en entrenamiento contra el de datos que la red nunca vio. Una brecha grande indica memorización.
- **¿Hay fuga de datos?** Revisar que no haya filas duplicadas o casi idénticas entre entrenamiento y prueba, ni variables que “delaten” la respuesta, ni normalización calculada con todo el dataset.
- **¿Los datos nuevos se parecen a los de entrenamiento?** Puede haber un cambio de distribución (otra iluminación, otros usuarios, otro rango de valores).
- **¿La métrica engaña?** Con clases desbalanceadas, “muy preciso” puede significar que siempre responde la clase mayoritaria. Mirar matriz de confusión, precisión y recall.
- **¿El modelo es adecuado para el problema y el tamaño del dataset?** Comparar contra una línea base simple (modelo lineal o red pequeña).
- **¿El preprocesamiento es el mismo en entrenamiento y en uso real (escalado, redimensionado, orden de columnas)?**

**2.**

- Curvas de pérdida y exactitud por época, de entrenamiento y validación, en la misma gráfica, para ver si divergen (sobreajuste) o si ambas se estancan.
- Evaluación en un conjunto de prueba separado desde el inicio, con matriz de confusión y ejemplos concretos de errores, junto con la explicación de cómo se hizo la partición aleatoria, por usuario, por fecha para descartar fuga de datos.

**3.**

- “Tienen una tabla con cinco columnas numéricas y cerca de mil filas. ¿Con qué modelo empezarían y qué los haría cambiar a otro? ¿Qué línea base compararon?”
- “Si aplanan una imagen y se la dan a un MLP, ¿qué información pierden y qué les da una CNN que el MLP no tiene? ¿Cómo cambiaría el número de parámetros?”
- “Muéstrenme un caso de sus datos que un perceptrón no pueda separar y expliquen qué parte del MLP lo resuelve. ¿Cómo comprobaron que la red no solo memorizó?”