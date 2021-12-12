import streamlit as st
import pandas as pd  						# Para la manipulación y análisis de los datos

from IA_functions_CJ import *
from IA_variables import *					#Variables definidas en otro documento

from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------------------
# INICIO DE REGRESIÓN LOGÍSTICA
# --------------------------------------------------------------------------------
def set_regresion():
	st.title('Regresión logística')
	img_distance = 'https://blog.dormakaba.com/tachyon/2019/08/58.jpg?resize=1024%2C683&zoom=1'
	st.image(img_distance, use_column_width='always', caption=quote3)

	st.write(info_regresion_logistia)
	st.subheader("¿Qué datos te gustaría estudiar?")
	data_file = set_archivoRL() 
	
	if data_file is not None:
		data_file = data_file.replace({'M': 0, 'B': 1})
		matrizMapaCJ(data_file)
		st.subheader("Selección de variables predictoras")
		st.write(info_vars_predictoras)
		selecVars = seleccionaVars(data_file) #X
		varsPredictoras = np.array(data_file[selecVars])
		varClase = seleccionaVarClase(data_file) #Y
		regresionL(varsPredictoras, varClase)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA SELECCIONAR LA VARIABLE A PREDECIR
# --------------------------------------------------------------------------------
def seleccionaVarClase(data_file):

	st.subheader("Selección de variable clase")
	st.write(info_vars_apronosticar)
	var = st.selectbox(
     'Selecciona la variable a predecir',
     data_file.columns)
	varClase = np.array(data_file[var])
	#st.dataframe(varClase)
	return varClase
# --------------------------------------------------------------------------------
# FUNCIÓN PARA GENERAR LAS PREDICCIONES
# --------------------------------------------------------------------------------
def regresionL(X,Y):

	porcentaje = st.slider("Porcentaje para la prueba: ", min_value=20, max_value=30, value=20)


	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                           	random_state=0,
                                                                       	    test_size = (int(porcentaje)/100),
                                                                   	        shuffle = True)
	Clasificacion = linear_model.LogisticRegression()
	Clasificacion.fit(X_train, Y_train)

	Y_Clasificacion = Clasificacion.predict(X_validation)
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


# Reporte de clasificación
	generar_reporte = st.checkbox('Generar reporte de clasificación')	
	if generar_reporte:
		st.subheader('Reporte de clasificación')
		#{:.2f}".format(numero)
		st.write("Exactitud: "+str("{:.4f}".format(Clasificacion.score(X_validation, Y_validation).round(6)*100))+" %")
		precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
		st.write("Precisión: "+ str(precision)+ " %")
		st.write("Tasa de error: "+str("{:.4f}".format((1-Clasificacion.score(X_validation, Y_validation))*100))+" %")
		sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
		st.write("Sensibilidad: "+ str(sensibilidad)+ " %")
		especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
		st.write("Especificidad: "+ str(especificidad)+" %")
	                            
		st.subheader('Modelo de clasificación: ')
		# Ecuación del modelo
		st.latex(r"p=\frac{1}{1+e^{-(a+bX)}}")

# --------------------------------------------------------------------------------
# FUNCIÓN PARA SUBIR ARCHIVOS 
# --------------------------------------------------------------------------------
#La siguiente función la utilizan todos los algoritmos, 
# ya que sirve para subir los archivos y mostrarlos
def set_archivoRL(): #Función para subir un archivo 
	data_file = st.file_uploader("Selecciona un archivo [.CSV]",
								type = ["csv"], key='RegresionL') #El usuario proporciona un archivo de extensión CSV
	if data_file is not None:
			data_file = pd.read_csv(data_file) 	 
			st.subheader('Visualización de los datos')    
			st.dataframe(data_file) #Muestra la data
			return data_file #Regresa el archivo subido 