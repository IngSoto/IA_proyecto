import streamlit as st
import pandas as pd  						# Para la manipulación y análisis de los datos

from IA_functions_CJ import *
from IA_functions_RL import *

from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
from sklearn import model_selection 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------------
# INICIO DE ÁRBOLES DE DECISIÓN
# --------------------------------------------------------------------------------

def set_arboles():
	st.title('Árboles de Decisión (Pronóstico y Clasificación)')
	img_distance = 'https://www.naept.com/wp-content/uploads/2020/08/tree.jpg'
	st.image(img_distance, use_column_width='always', caption=quote1)

	#st.write(info_regresion_logistia)

	st.subheader("¿Qué datos te gustaría estudiar?")
	data_file = set_archivoAD() 
	
	if data_file is not None:
		matrizMapaCJ(data_file)
		st.subheader("Selección de variables predictoras")
		st.write(info_vars_predictoras)
		selecVars = seleccionaVars(data_file) #X
		varsPredictoras = np.array(data_file[selecVars])
		varClase = seleccionaVarClase(data_file) #Y


		tipoArbol = st.radio("",
					("Árbol de decisión: Regresión", 
					 "Árbol de decisión: Clasificación"))

		if tipoArbol == "Árbol de decisión: Regresión":
			st.subheader("Árbol de decisión: Regresión")
			st.write(info_arbol_regresion)
			arbolAlgoritmos(data_file, varsPredictoras, varClase, selecVars, 1)

		if tipoArbol == "Árbol de decisión: Clasificación":
			st.subheader("Árbol de decisión: Clasificación")
			#st.write(info_arbol_regresion)
			arbolAlgoritmos(data_file, varsPredictoras, varClase, selecVars, 2)


# --------------------------------------------------------------------------------
# FUNCIÓN PARA APLICAR EL ALGORITMO DE ÁRBOL DE REGRESIÓN
# --------------------------------------------------------------------------------

def arbolAlgoritmos(data_file,X,Y, varsX, tipoAlgoritmo):
	porcentaje = st.slider("Porcentaje para la prueba: ", min_value=20, max_value=30, value=20)

	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                           	random_state=1234,
                                                                       	    test_size = (int(porcentaje)/100),
                                                                   	        shuffle = True)



	st.subheader("Selecciona las características del árbol")
	st.write(info_max_depth)
	max_profundidad = st.number_input("Max_depth: ", min_value=1, value=8)
	st.write(info_min_samples_split)
	min_muestras_corte = st.number_input("Min_samples_split: ", min_value=1, value=2)
	st.write(info_min_samples_leaf)
	min_muestras_leaf = st.number_input("Min_samples_leaf: ", min_value=1, value=1)

	if tipoAlgoritmo == 1: ## REGRESIÓN
		pronosticoAD = DecisionTreeRegressor(max_depth=max_profundidad, min_samples_split=min_muestras_corte, min_samples_leaf=min_muestras_leaf,random_state=0)
	elif tipoAlgoritmo == 2: ## CLASIFICACIÓN
		pronosticoAD = DecisionTreeClassifier(max_depth=max_profundidad, min_samples_split=min_muestras_corte, min_samples_leaf=min_muestras_leaf,random_state=0)


	pronosticoAD.fit(X_train, Y_train)

	st.subheader('Importancia de las variables')
	st.write(info_importancia)
	Importancia = pd.DataFrame({'Variable': varsX,
		'Importancia': pronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
	st.table(Importancia)

	if tipoAlgoritmo == 2: ## CLASIFICACIÓN - Matriz de confusión

		Y_Clasificacion = pronosticoAD.predict(X_validation)
		Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                   Y_Clasificacion, 
                                   rownames=['Real'], 
                                   colnames=['Clasificación']) 

		st.subheader("Matiz de confusión")
		st.write(info_matriz_confusion)
		#st.table(Matriz_Clasificacion)
		col1, col2 = st.columns(2)
		col1.info('Verdaderos Positivos: '+str(Matriz_Clasificacion.iloc[1,1]))
		col2.info('Falsos Negativos: '+str(Matriz_Clasificacion.iloc[1,0]))
		col2.info('Verdaderos Negativos: '+str(Matriz_Clasificacion.iloc[0,0]))
		col1.info('Falsos Positivos: '+str(Matriz_Clasificacion.iloc[0,1]))


	Y_Pronostico = pronosticoAD.predict(X_train)

	st.subheader("Obtención de los parámetros del modelo")
	st.write('Criterio: \n', pronosticoAD.criterion)
	st.write("MAE: %.4f" % mean_absolute_error(Y_train, Y_Pronostico))
	st.write("MSE: %.4f" % mean_squared_error(Y_train, Y_Pronostico))
	st.write("RMSE: %.4f" % mean_squared_error(Y_train, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
	st.write('Score: %.4f' % r2_score(Y_train, Y_Pronostico))

	st.subheader("Árbol de decisión")
	st.write(info_arbol_grafica)
	figura = plt.figure(figsize=(60,60))  
	plot_tree(pronosticoAD, feature_names = varsX)
	st.pyplot(figura)

	st.subheader("Árbol de decisión en formato de texto")
	Reporte = export_text(pronosticoAD, feature_names = varsX)
	st.text(Reporte)
	
# --------------------------------------------------------------------------------
# FUNCIÓN PARA SUBIR ARCHIVOS 
# --------------------------------------------------------------------------------
#La siguiente función la utilizan todos los algoritmos, 
# ya que sirve para subir los archivos y mostrarlos
def set_archivoAD(): #Función para subir un archivo 
	data_file = st.file_uploader("Selecciona un archivo [.CSV]",
								type = ["csv"], key='ArbolesD') #El usuario proporciona un archivo de extensión CSV
	if data_file is not None:
			data_file = pd.read_csv(data_file) 	 
			st.subheader('Visualización de los datos')    
			st.dataframe(data_file) #Muestra la data
			return data_file #Regresa el archivo subido 