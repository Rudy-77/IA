Problema 1: Los 3 caníbales y los 3 monjes
Tenemos 3 monjes y 3 caníbales que quieren cruzar un río en una canoa que solo aguanta 2 personas. La regla es que en ninguna orilla pueden haber más caníbales que monjes, porque si eso pasa, se comen a los monjes. Si no hay monjes en una orilla, no importa cuántos caníbales haya.
Solución:

1. Cruzan 2 caníbales.
2. Regresa 1 caníbal.
3. Cruzan los 2 caníbales que quedaban.
4. Regresa 1 caníbal.
5. Cruzan 2 monjes.
6. Regresan 1 monje y 1 caníbal.
7. Cruzan 2 monjes.
8. Regresa 1 caníbal.
9. Cruzan 2 caníbales.
10. Regresa 1 caníbal.
11. Cruzan los últimos 2 caníbales.

Problema 2: El laberinto
Problema 3: Conteo de islas en una matriz n×m
Aquí tenemos una matriz llena de 1's (tierra) y 0's (agua), y queremos saber cuántas islas hay. Una isla es un grupo de 1's que están pegados entre sí (arriba, abajo, izquierda o derecha).
La forma más sencilla de resolverlo es recorrer toda la matriz celda por celda. Cada vez que encontramos un 1 que no hemos visitado antes, sabemos que es una isla nueva, así que sumamos 1 al contador. Luego, desde esa celda, exploramos todos sus vecinos que también sean tierra (esto se puede hacer con DFS o BFS), y los vamos marcando como visitados para no volver a contarlos después.
Seguimos recorriendo la matriz así hasta terminar. Al final, el contador nos dice cuántas islas había en total. En pocas palabras, cada vez que "pisamos" tierra sin visitar, es una isla nueva, y con la búsqueda marcamos toda esa isla para no contarla dos veces.
