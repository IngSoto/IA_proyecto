# --------------------------------------------------------------------------------
# VARIABLES DE USO GENERAL
# --------------------------------------------------------------------------------

info_tipoUsuario = '''
## ¿Qué tipo de usuario eres?
'''


# --------------------------------------------------------------------------------
# VARIABLES PARA EL ALGORITMO REGLAS DE ASOCIACIÓN
# --------------------------------------------------------------------------------

info_reglas_asociacion = '''
Las reglas de asociación encuentran relaciones interesantes entre grandes conjuntos de elementos de datos. Esta regla muestra la frecuencia con la que se produce un conjunto de elementos en una transacción. Un ejemplo típico es el análisis basado en el mercado.

El análisis basado en el mercado es una de las técnicas clave que se utilizan para mostrar asociaciones entre elementos. Permite a los minoristas identificar las relaciones entre los artículos que las personas compran juntas con frecuencia.

Dado un conjunto de transacciones, podemos encontrar reglas que predecirán la ocurrencia de un artículo en función de las ocurrencias de otros artículos en la transacción.
'''

info_visualizar_datafile = '''
1) NaN indica que ese elemento no fue adquirido en esa transacción.
'''

info_procesar = '''
Antes de ejecutar el algoritmo es recomendable observar la distribución de la frecuencia de los elementos.
'''

info_implementar = '''
Para grandes conjuntos de datos, puede haber cientos de elementos en cientos de miles de transacciones. El algoritmo Apriori intenta extraer reglas para cada combinación posible de elementos. 
'''

info_soporte = '''
El soporte se refiere a la popularidad predeterminada de un artículo y se puede calcular encontrando el número de transacciones que contienen un artículo en particular dividido por el número total de transacciones. 
'''

info_confianza = '''
La confianza se refiere a la probabilidad de que también se compre un artículo B si se compra el artículo A. Se puede calcular encontrando el número de transacciones en las que A y B se compran juntos, dividido por el número total de transacciones en las que se compra A.
'''

info_elevacion = '''
La elevación indica el nivel de relación (aumento de probabilidad) entre el antecedente y consecuente de la regla. La elevación (A -> B) se puede calcular dividiendo la confianza(A -> B) dividido por el soporte(B)
'''

# --------------------------------------------------------------------------------
# VARIABLES PARA EL ALGORITMO DE MÉTRICAS DE DISTANCIA
# --------------------------------------------------------------------------------

info_metricas_distancia = '''
Las métricas de distancia son una parte clave de varios algoritmos de aprendizaje automático. Estas métricas de distancia se utilizan tanto en el aprendizaje supervisado como no supervisado, generalmente para calcular la similitud entre puntos de datos.

Una métrica de distancia efectiva mejora el rendimiento de nuestro modelo de aprendizaje automático, ya sea para tareas de clasificación o agrupación.
'''

info_distancias = '''
## ¿Qué distancia te gustaría calcular?
#### 1. Distancia Euclidiana.
Es una medida de distancia entre un par de muestras p y q en un espacio de características n-dimensional.
#### 2. Distancia de Chebyshev. 
Es el valor máximo absoluto de las diferencias entre las coordenadas de un par de elementos.
#### 3. Distancia de Manhattan.
Calcula la distancia entre dos puntos en una ruta similar a una cuadrícula (información geoespacial).
#### 4. Distancia de Minkowsky.
Es una distancia entre dos puntos en un espacio n-dimensional. Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev.
'''

info_estandarizacion = '''
#### Recuerda, experto😎:
Las distancias dependen en gran medida del número de constructos y del rango de clasificación. Para poder comparar distancias entre cuadrículas de diferente tamaño y rango de clasificación, es deseable una estandarización.

La estandarización de un conjunto de datos es un requisito común para muchos estimadores de aprendizaje automático, ya que pueden comportarse mal si las características individuales no se ven más o menos como datos estándar distribuidos normalmente. 

Si una característica tiene una varianza que es órdenes de magnitud mayor que otras, podría dominar la función objetivo y hacer que el estimador no pueda aprender de otras características correctamente como se esperaba. 
'''


info_parametrop = '''
Asimismo, toma en consideración que para la distancia Minkowski se debe añadir un parámetro llamado p, el cual corresponde al parámetro lambda, que es el orden para calcular la distancia de tres formas diferentes
	λ = 1, distancia de Manhattan.
	λ = 2, distancia Euclidiana.
	λ = 3, distancia de Chebyshev.
Dependiendo del valor insertado, proporcionará una aproximación curvilínea diferente.
'''

# --------------------------------------------------------------------------------
# VARIABLES PARA CLUSTERING JERÁRQUICO
# --------------------------------------------------------------------------------

info_cluster = '''
La agrupación en clústeres es una técnica de aprendizaje automático cuyo objetivo es agrupar los puntos de datos que tienen propiedades y/o características similares, mientras que los puntos de datos en diferentes grupos deben tener propiedades y/o características muy poco convencionales.

El clustering es muy importante ya que determina la agrupación intrínseca entre los datos no etiquetados presentes. No hay criterios para una buena agrupación. Depende del usuario, cuál es el criterio que puede utilizar para satisfacer su necesidad. 
'''

info_matriz_correlacion = '''
Una matriz de correlación es una tabla que muestra los coeficientes de correlación para diferentes variables. La matriz muestra la correlación entre todos los posibles pares de valores en una tabla. Es una herramienta poderosa para resumir un gran conjunto de datos e identificar y visualizar patrones en los datos dados.
'''

info_mapa_calor = '''
Un mapa de calor es una representación gráfica de datos donde cada valor de una matriz se representa como un color.
'''

info_clustering_jerarquico = '''
El agrupamiento jerárquico, también conocido como análisis de agrupamiento jerárquico, es un algoritmo que agrupa objetos similares en grupos llamados agrupamientos. El punto final es un conjunto de clústeres, donde cada clúster es distinto de los demás y los objetos dentro de cada clúster son muy similares entre sí.
'''


info_dendograma = '''
Un dendrograma es un diagrama que muestra la relación jerárquica entre objetos. Por lo general, se crea como resultado de la agrupación jerárquica. El uso principal de un dendrograma es encontrar la mejor manera de asignar objetos a grupos.
'''

info_cluter_registro = '''
Estos clústers formados agrupan a todos aquellos datos con mayor relación, tomando como base la
matriz estandarizada de los datos.
'''

info_centroides = '''
De cada clúster se obtiene el promedio de los valores que tienen sus registros respecto a las variables consideradas
para el proceso de clusterización. Con base en estos centroides, es posible que un especialista genera las respectivas conclusiones.
'''

# --------------------------------------------------------------------------------
# VARIABLES PARA CLUSTERING PARTICIONAL
# --------------------------------------------------------------------------------

info_clustering_particional= '''
La agrupación en clústeres particional (o agrupación en clústeres de particiones) son métodos de agrupación que se utilizan para clasificar las observaciones, dentro de un conjunto de datos, en varios grupos en función de su similitud. 

El algoritmo k-means es un método de agrupamiento que divide un conjunto de n observaciones en k grupos distintos gracias a valores medios. Pertenece al ámbito de los algoritmos no supervisados, ya que las n observaciones no cuentan con una etiqueta que nos diga de qué grupo es cada dato, siendo los datos agrupados según sus propiedades o características.
'''

info_metodo_codo = '''
La idea básica de los algoritmos de clustering es la minimización de la varianza intra-cluster y la maximización de la varianza inter-cluster. Es decir, queremos que cada observación se encuentre muy cerca a las de su mismo grupo y los grupos lo más lejos posible entre ellos.

El método del codo utiliza la distancia media de las observaciones a su centroide. Es decir, se fija en las distancias intra-cluster. Cuanto más grande es el número de clusters k, la varianza intra-cluster tiende a disminuir. Cuanto menor es la distancia intra-cluster mejor, ya que significa que los clústers son más compactos. El método del codo busca el valor k que satisfaga que un incremento de k, no mejore sustancialmente la distancia media intra-cluster.
'''
# --------------------------------------------------------------------------------
# VARIABLES PARA REGRESIÓN LOGÍSTICA
# --------------------------------------------------------------------------------

info_regresion_logistia = '''
La regresión logística resulta útil para los casos en los que se desea predecir la presencia o ausencia de una característica o resultado según los valores de un conjunto de predictores. Es similar a un modelo de regresión lineal pero está adaptado para modelos en los que la variable dependiente es dicotómica. Los coeficientes de regresión logística pueden utilizarse para estimar la razón de probabilidad de cada variable independiente del modelo. La regresión logística se puede aplicar a un rango más amplio de situaciones de investigación que el análisis discriminante.
'''


info_vars_predictoras = '''
Las variables predictoras representarán al conjunto de las X, que son las variables independientes de nuestro modelo.
'''

info_vars_apronosticar = '''
Las variables a pronosticar representarán al conjunto de las Y, que son las variables dependientes de nuestro modelo.
'''

info_matriz_confusion = '''
La matriz de confusión muestra el número de clases pronosticadas y el número de clasificaciones correctas.
'''

# --------------------------------------------------------------------------------
# VARIABLES PARA ÀRBOL DE REGRESIÓN
# --------------------------------------------------------------------------------

info_arbol_regresion = '''
La regresión del árbol de decisión observa las características de un objeto y entrena un modelo en la estructura de un árbol para predecir datos en el futuro.
'''

info_max_depth = '''
max_depth se refiere a la profundidad máxima hasta la cual llegará el árbol. Por defecto se tiene el máximo
de profundidad posible.
'''

info_min_samples_split = '''
min_samples_split se refiere a la cantidad mínima de resultados que deben de existir para que se realice una división en las
muestras. Por defecto se tiene un valor de 2.
'''

info_min_samples_leaf = '''
min_samples_leaf la cantidad mínima de elementos que debe de haber en cada nodo hoja del árbol. Por defecto se tiene un valor de 1.
'''

info_importancia = '''
Se presentan el nombre de la variable y su respectivo valor de importancia.
'''

info_arbol_grafica = '''
Árbol con los respectivos nombres de las variables. Se muestra cada uno de los nodos que conforman al árbol, con
el respectivo criterio o regla, error cuadrático, número de elementos que lo conforman y el valor promedio.
'''
# --------------------------------------------------------------------------------
# QUOTES
# --------------------------------------------------------------------------------

quote1 = '''
La clave de la inteligencia artificial siempre ha sido la representación. – Jeff Hawkins
'''

quote2 = '''
Visualizo una época en la que (los humanos) seremos a los robots lo que los perros son para nosotros. – Claude Shannon
'''

quote3 = '''
Estamos en un coche yendo hacia el futuro utilizando sólo nuestro espejo retrovisor. – Herbert Marshall Mcluhan
'''
quote4 = '''
Habrá seres humanos con minirrobots en el cerebro. – Raymond Kurzweil
'''
quote5 = '''
Las máquinas son objetos, que han sido construidos para vencer la resistencia del mundo, resistencia con la que choca el trabajo. – Vilém Flusser
'''

# --------------------------------------------------------------------------------
# SOBRE ALGORTHMIA
# --------------------------------------------------------------------------------

info_algorithmia = '''
El nombre ALGORITHMIA es un juego de palabras que combina ALGORITHM (de algoritmo) y IA (de Inteligencia Artificial). En este entorno puedes escoger diferentes algoritmos que te permitirán examinar, estudiar y aplicar distintos algoritmos de Inteligencia Artificial. Entre ellos están: Reglas de Asociación, Métricas de distancia, Clusteirng, Regresión Logística y Árboles de Decisión. 
'''




















