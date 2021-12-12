import streamlit as st
import numpy as np							# Para crear vectores y matrices n dimensionales
import pandas as pd  						# Para la manipulación y análisis de los datos

#Métricas de distancia
from scipy.spatial.distance import cdist 	#Para el cálculo de distancias
from scipy.spatial import distance

from PIL import Image 						#Para importar imágenes
from IA_variables import *					#Variables definidas en otro documento
from IA_functions_principals import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler  #Para estandarizar los datos



# --------------------------------------------------------------------------------
# INICIO DE MÉTRICAS DE DISTANCIA
# --------------------------------------------------------------------------------
def set_metricasDistancia():
	st.title('Métricas de distancia')
	img_distance = 'https://datascience.foundation/backend/web/uploads/blog/Distance%20Train.jpg'
	st.image(img_distance, use_column_width='always', caption=quote5)

	st.write(info_metricas_distancia)

	st.subheader("¿Qué datos te gustaría estudiar?")
	data_file = set_archivoMD() 
	if data_file is not None:
		menu_metricasDistancia(data_file)
# --------------------------------------------------------------------------------
# MENÚ PARA SELECCIONAR UNA MÉTRICA DE DISTANCIA
# --------------------------------------------------------------------------------
def menu_metricasDistancia(data_file):
	#if data_file is not None:

	st.write(info_tipoUsuario, unsafe_allow_html=True)
	tipoUsuario = st.radio("",
							("Estudiante🤓", 
							 "Experto😎"))

	st.write(info_distancias, unsafe_allow_html=True)
	medidasDistancia = st.radio("",
								("Euclidiana",
								"Chebyshev",
								"Manhattan",
								"Minkowski"))

	standard = StandardScaler()
	MEstandarizada = standard.fit_transform(data_file)

	if (medidasDistancia == "Euclidiana" and tipoUsuario == "Estudiante🤓"):
		distanciaEuclidiana(data_file)
	elif (medidasDistancia == "Euclidiana" and tipoUsuario == "Experto😎"):
		distanciaEuclidianaXP(MEstandarizada)

	elif (medidasDistancia == "Chebyshev" and tipoUsuario == "Estudiante🤓"):
		distanciaChebyshev(data_file)
	elif (medidasDistancia == "Chebyshev" and tipoUsuario == "Experto😎"):
		distanciaChebyshevXP(MEstandarizada)

	elif (medidasDistancia == "Manhattan" and tipoUsuario == "Estudiante🤓"):
		distanciaManhattan(data_file)
	elif (medidasDistancia == "Manhattan" and tipoUsuario == "Experto😎"):
		distanciaManhattanXP(MEstandarizada)

	elif (medidasDistancia == "Minkowski" and tipoUsuario == "Estudiante🤓"):
		distanciaMinkowski(data_file)
	elif (medidasDistancia == "Minkowski" and tipoUsuario == "Experto😎"):
		distanciaMinkowskiXP(MEstandarizada)
# --------------------------------------------------------------------------------
# FUNCIONES PARA EL ALGORITMO: MÉTRICAS DE DISTANCIA
# --------------------------------------------------------------------------------

# ------------------------------------------------
# DISTANCIA EUCLIDIANA USUARIO ESTUDIANTES
# ------------------------------------------------
def distanciaEuclidiana(data_file):
	st.subheader("Distancia Euclidiana")
	st.write("#### Matriz de distancia entre todos los objetos:")
	DatEuclidiana = cdist(data_file, data_file, metric='euclidean')
	MEuclidiana = pd.DataFrame(DatEuclidiana)
	st.dataframe(MEuclidiana)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = data_file.iloc[st.slider('Objeto 1:', 0, len(MEuclidiana)-1)]
	obj2 = data_file.iloc[st.slider('Objeto 2:', 0, len(MEuclidiana)-1)]
	
	distanciaEuclidiana = distance.euclidean(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaEuclidiana))

# ------------------------------------------------
# DISTANCIA EUCLIDIANA USUARIO EXPERTO
# ------------------------------------------------
def distanciaEuclidianaXP(MEstandarizada):
	st.subheader("Distancia Euclidiana Estandarizada")
	st.markdown(info_estandarizacion)
	st.write("#### Matriz de distancia estandarizada entre todos los objetos:")

	DatEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
	MEuclidiana = pd.DataFrame(DatEuclidiana)
	st.dataframe(MEuclidiana.round(2))

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = MEstandarizada[st.slider('Objeto 1:', 0, len(MEuclidiana)-1)]
	obj2 = MEstandarizada[st.slider('Objeto 2:', 0, len(MEuclidiana)-1)]
	
	distanciaEuclidiana = distance.euclidean(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaEuclidiana))

# ------------------------------------------------
# DISTANCIA CHEBYSHEV USUARIO ESTUDIANTES
# ------------------------------------------------ 
def distanciaChebyshev(data_file):
	st.subheader("Distancia Chebyshev")
	st.write("#### Matriz de distancia entre todos los objetos:")
	DatChebyshev = cdist(data_file, data_file, metric='chebyshev')
	MChebyshev = pd.DataFrame(DatChebyshev)
	st.dataframe(MChebyshev)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = data_file.iloc[st.slider('Objeto 1:', 0, len(MChebyshev)-1)]
	obj2 = data_file.iloc[st.slider('Objeto 2:', 0, len(MChebyshev)-1)]
	
	distanciaChebyshev = distance.chebyshev(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaChebyshev))

# ------------------------------------------------
# DISTANCIA CHEBYSHEV USUARIO EXPERTO 
# ------------------------------------------------ 
def distanciaChebyshevXP(MEstandarizada):
	st.subheader("Distancia Chebyshev Estandarizada")
	st.markdown(info_estandarizacion)
	st.write("#### Matriz de distancia estandarizada entre todos los objetos:")
	DatChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
	MChebyshev = pd.DataFrame(DatChebyshev)
	st.dataframe(MChebyshev)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = MEstandarizada[st.slider('Objeto 1:', 0, len(MChebyshev)-1)]
	obj2 = MEstandarizada[st.slider('Objeto 2:', 0, len(MChebyshev)-1)]
	
	distanciaChebyshev = distance.chebyshev(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaChebyshev))

# ------------------------------------------------
# DISTANCIA MANHATTAN USUARIO ESTUDIANTES
# ------------------------------------------------ 

def distanciaManhattan(data_file):	
	st.subheader("Distancia Manhattan")
	st.write("#### Matriz de distancia entre todos los objetos:")
	DatManhattan = cdist(data_file, data_file, metric='cityblock')
	MManhattan = pd.DataFrame(DatManhattan)
	st.dataframe(MManhattan)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = data_file.iloc[st.slider('Objeto 1:', 0, len(MManhattan)-1)]
	obj2 = data_file.iloc[st.slider('Objeto 2:', 0, len(MManhattan)-1)]
	
	distanciaManhattan = distance.cityblock(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaManhattan))

# ------------------------------------------------
# DISTANCIA MANHATTAN USUARIO EXPERTO 
# ------------------------------------------------ 

def distanciaManhattanXP(MEstandarizada):	
	st.subheader("Distancia Manhattan Estandarizada")
	st.markdown(info_estandarizacion)
	st.write("#### Matriz de distancia estandarizada entre todos los objetos:")
	DatManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
	MManhattan = pd.DataFrame(DatManhattan)
	st.dataframe(MManhattan)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = MEstandarizada[st.slider('Objeto 1:', 0, len(MManhattan)-1)]
	obj2 = MEstandarizada[st.slider('Objeto 2:', 0, len(MManhattan)-1)]
	
	distanciaManhattan = distance.cityblock(obj1, obj2)

	st.write("#### La distancia entre los objetos es: "+str(distanciaManhattan))

# ------------------------------------------------
# DISTANCIA MANHATTAN USUARIO ESTUDIANTES
# ------------------------------------------------
def distanciaMinkowski(data_file):	
	st.subheader("Distancia Minkowski")
	st.write("#### Matriz de distancia entre todos los objetos con p = 1.5:")
	DatMinkowski = cdist(data_file, data_file, metric='minkowski', p=1.5) #ojo con el ajuste
	MMinkowski = pd.DataFrame(DatMinkowski)
	st.dataframe(MMinkowski)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = data_file.iloc[st.slider('Objeto 1:', 0, len(MMinkowski)-1)]
	obj2 = data_file.iloc[st.slider('Objeto 2:', 0, len(MMinkowski)-1)]
	
	distanciaMinkowski = distance.minkowski(obj1, obj2, p=1.5)

	st.write("#### La distancia entre los objetos es: "+str(distanciaMinkowski))


# ------------------------------------------------
# DISTANCIA MANHATTAN USUARIO EXPERTO
# ------------------------------------------------ 
def distanciaMinkowskiXP(MEstandarizada):	
	st.subheader("Distancia Minkowski Estandarizada")
	st.markdown(info_estandarizacion)
	st.markdown(info_parametrop)
	parametro_p = st.number_input('Por favor, inserta el orden para calcular las distancias (p):', 
								min_value=1.0,
								max_value=3.0)
	st.write('El orden actual es de ', parametro_p)

	##number = st.number_input('Insert a number')
	##st.write('The current number is ', number)

	st.write("#### Matriz de distancia estandarizada entre todos los objetos:")
	DatMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=parametro_p) #ojo con el ajuste
	MMinkowski = pd.DataFrame(DatMinkowski)
	st.dataframe(MMinkowski)

	st.write("Distancia entre dos objetos:")
	
	obj1 = []
	obj2 = []
	
	obj1 = MEstandarizada[st.slider('Objeto 1:', 0, len(MMinkowski)-1)]
	obj2 = MEstandarizada[st.slider('Objeto 2:', 0, len(MMinkowski)-1)]
	
	distanciaMinkowski = distance.minkowski(obj1, obj2, p=parametro_p)

	st.write("#### La distancia entre los objetos es: "+str(distanciaMinkowski))

# --------------------------------------------------------------------------------
# FUNCIÓN PARA SUBIR ARCHIVOS 
# --------------------------------------------------------------------------------
#La siguiente función la utilizan todos los algoritmos, 
# ya que sirve para subir los archivos y mostrarlos
def set_archivoMD(): #Función para subir un archivo 
	data_file = st.file_uploader("Selecciona un archivo [.CSV]",
								type = ["csv"], key = 'MetricasD') #El usuario proporciona un archivo de extensión CSV
	if data_file is not None:
			data_file = pd.read_csv(data_file) 	 
			st.subheader('Visualización de los datos')    
			st.dataframe(data_file) #Muestra la data
			return data_file #Regresa el archivo subido 






