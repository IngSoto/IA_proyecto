
import streamlit as st
from IA_variables import *					#Variables definidas en otro documento

import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib

#Clúster Jerárquico
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import neattext.functions as nfx
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
# --------------------------------------------------------------------------------
# INICIO DE CLUSTERING JERÁRQUICO 
# --------------------------------------------------------------------------------

def set_clustering():
	st.title('Clustering')
	img_cj = 'https://miro.medium.com/max/2000/0*pX0ixk2x9VAaK8-a.jpeg'
	st.image(img_cj, use_column_width='always', caption=quote2)
	st.write(info_cluster)
	st.subheader("¿Qué datos te gustaría estudiar?")
	data_file = set_archivoCJ() 
	if data_file is not None:
		matrizMapaCJ(data_file)
		st.subheader("Selección de variables")
		variables = seleccionaVars(data_file)
		MVariables = np.array(data_file[variables])
		st.dataframe(MVariables)
		estandarizar = StandardScaler() # Se instancia el objeto StandardScaler o MinMaxScaler 
		MEstandarizada = estandarizar.fit_transform(MVariables)   # Se calculan la media y desviación y se escalan los datos

		metrica = st.selectbox(
     		'Selecciona la métrica de distancia que utilizarás',
     		('euclidean','chebyshev','cityblock','minkowski'))

		tipoCluster = st.radio("",
					("Clúster Jerárquico", 
					 "Clúster Particional"))

		if tipoCluster == "Clúster Jerárquico":
			st.subheader("Clúster Jerárquico Ascendente")
			st.write(info_clustering_jerarquico)
			aplicarJerarquicoA(MEstandarizada, metrica)
			seleccionarClusteres(data_file, MEstandarizada, metrica, variables)

		if tipoCluster == "Clúster Particional":
			st.subheader("Clúster Particional K-Means")
			st.write(info_clustering_particional)
			localizador = metodoCodo(MEstandarizada)
			seleccionarClusteresPart(data_file, MEstandarizada, variables, localizador)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA MOSTAR LA MATRIZ Y EL MAPA DE CALOR
# --------------------------------------------------------------------------------
def matrizMapaCJ(data_file):
	st.subheader("Matriz de correlaciones")
	st.write(info_matriz_correlacion)
	corrData = data_file.corr(method='pearson')
	st.dataframe(corrData) #Muestra la data

	st.subheader("Mapa de calor")
	st.write(info_mapa_calor)
	grafica = plt.figure(figsize=(14,7))
	MatrizInf = np.triu(corrData)
	sns.heatmap(corrData, cmap='RdBu_r', annot=True, mask=MatrizInf)
	st.pyplot(grafica)
# --------------------------------------------------------------------------------
# FUNCIÓN PARA SELECCIONAR LAS VARIABLES
# --------------------------------------------------------------------------------
def seleccionaVars(data_file):
	
	variables = st.multiselect(
     'Selecciona las variables por considerar',
     data_file.columns,
     default = data_file.columns.all())

	return variables 
# --------------------------------------------------------------------------------
# FUNCIÓN PARA GENERAR EL DENDOGRAMA Y HACER UN CORTE
# --------------------------------------------------------------------------------
def aplicarJerarquicoA(MEstandarizada, metrica):
	
	st.subheader("Dendograma: Clusteres creados")
	st.write(info_dendograma)
	grafica = plt.figure(figsize=(10, 7))
	plt.title("Clustering Jerárquico Ascendente")
	plt.xlabel('Observaciones')
	plt.ylabel('Distancia')
	Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric=metrica))
	
	nivel = st.slider("Selecciona el nivel del dendograma: ", min_value=0.0, max_value=np.max(Arbol['dcoord']),step=0.1)
	plt.axhline(y=nivel, color='purple', linestyle='--') # Hace un corte en las ramas

	st.pyplot(grafica)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA SELECCIONAR UN NÚMERO DE CLUSTERES - JERÁRQUICO
# --------------------------------------------------------------------------------
def seleccionarClusteres(data_file,MEstandarizada, metrica, variables):
	num_clusteres = st.number_input('Número de clústeres',
									min_value=1,
									max_value=7)

	MJerarquico = AgglomerativeClustering(n_clusters=num_clusteres, linkage='complete', affinity=metrica)
	MJerarquico.fit_predict(MEstandarizada)

	data_clusters = data_file[variables]
	data_clusters['clusterH'] = MJerarquico.labels_
	st.subheader("Clusteres pertenecientes a cada registro")
	st.write(info_cluter_registro)
	st.dataframe(data_clusters)

	generarCentroides(data_clusters, num_clusteres)


# --------------------------------------------------------------------------------
# FUNCIÓN PARA SELECCIONAR UN NÚMERO DE CLUSTERES - PARTICIONAL
# --------------------------------------------------------------------------------	
def seleccionarClusteresPart(data_file, MEstandarizada, variables, num_clusteres):
	MParticional = KMeans(n_clusters=num_clusteres, random_state=0).fit(MEstandarizada)
	MParticional.predict(MEstandarizada)

	data_clusters = data_file[variables]
	data_clusters['clusterH'] = MParticional.labels_
	st.subheader("Clusteres pertenecientes a cada registro")
	st.write(info_cluter_registro)
	st.dataframe(data_clusters)	

	generarCentroides(data_clusters, num_clusteres)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA GENERAR CENTROIDES DE CADA CLUSTER
# --------------------------------------------------------------------------------
def generarCentroides(data_clusters, num_clusteres):
	CentroidesH = data_clusters.groupby('clusterH').mean()
	st.subheader("Centroides de los clústeres")
	st.write(info_centroides)
	st.dataframe(CentroidesH)

	resultados_text = list()

	#Generando conclusiones
	for i in range(num_clusteres):
		st.subheader("Clúster "+str(i+1))
		conclusion = st.text_area("Conclusiones sobre el clúster "+str(i+1) + ": ", "### Cluster " +str(i+1) + ":")
	
		resultados_text.append(conclusion)
	
	#Convertir lista en texto
	raw_text = ' '.join(map(str, resultados_text))

	prueba = nfx.clean_text(raw_text)
	st.download_button(label='🔥 Descargar conclusiones 🔥', data = prueba, file_name = 'ConclusionesClusters.txt')
                                       
# --------------------------------------------------------------------------------
# FUNCIÓN PARA GENERAR EL MÉTODO DEL CODO
# --------------------------------------------------------------------------------
def metodoCodo(MEstandarizada):
    st.subheader("Método del codo")
    st.write(info_metodo_codo)
    max_codo = st.number_input("Valor máximo para el método del codo: ",
    							min_value=2,
    							max_value=12)

    SSE = []
    for i in range(2, int(max_codo)):
        km = KMeans(n_clusters=i, random_state=0) #random_state toma una semilla para generar una posicion pseudoaleatoria de los centroides
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)

    grafica = plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_codo), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    st.pyplot(grafica)

    localizador = KneeLocator(range(2, max_codo), SSE, curve="convex", direction="decreasing")
    st.subheader("Localización del codo: " + str(localizador.elbow))
    return localizador.elbow


# --------------------------------------------------------------------------------
# FUNCIÓN PARA SUBIR ARCHIVOS 
# --------------------------------------------------------------------------------
#La siguiente función la utilizan todos los algoritmos, 
# ya que sirve para subir los archivos y mostrarlos
def set_archivoCJ(): #Función para subir un archivo 
	data_file = st.file_uploader("Selecciona un archivo [.CSV]",
								type = ["csv"]) #El usuario proporciona un archivo de extensión CSV
	if data_file is not None:
			data_file = pd.read_csv(data_file) 	 
			st.subheader('Visualización de los datos')    
			st.dataframe(data_file) #Muestra la data
			return data_file #Regresa el archivo subido 



