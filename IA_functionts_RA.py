import streamlit as st
import numpy as np							# Para crear vectores y matrices n dimensionales
import pandas as pd  						# Para la manipulación y análisis de los datos
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori
import neattext.functions as nfx

from PIL import Image 						#Para importar imágenes
from IA_variables import *					#Variables definidas en otro documento
from IA_functions_principals import *

# --------------------------------------------------------------------------------
# INICIO DE REGLAS DE ASOCIACIÓN
# --------------------------------------------------------------------------------
def set_reglasAsociacion():
	st.title('Reglas de Asociación')
	img_reglas_asociacion = 'https://www.metricser.com/wp-content/uploads/2020/10/img-algoritmos-inteligencia-artificial-001-600x337.jpg'
	st.image(img_reglas_asociacion, use_column_width='always', caption=quote4)
	st.write(info_reglas_asociacion)

	st.subheader("¿Qué datos te gustaría estudiar?")
	data_file = set_archivoRA() 
	if data_file is not None:
		st.write(info_visualizar_datafile)

		st.subheader("¿Qué te gustaría hacer?")
		st.write("#### 1. Procesamiento de datos")
		st.write(info_procesar)
		procesar = st.checkbox('Procesar los datos')
		if procesar:
		    procesarRA(data_file)

		st.write("#### 2. Aplicación del algoritmo apriori")
		st.write(info_implementar)
		implementar = st.checkbox('Aplicar el algoritmo apriori')		
		if implementar:
		    aplicarApriori(data_file)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA EXPLORAR Y PROCESAR LOS DATOS, GENERA GRÁFICA
# --------------------------------------------------------------------------------		
def procesarRA(data_file):
	st.subheader("Exploración de los datos")
	transacciones = data_file.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'

	#Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
	Lista = pd.DataFrame(transacciones)
	Lista['Frecuencia'] = 1

	#Se agrupa los elementos
	Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
	Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
	Lista = Lista.rename(columns={0 : 'Item'})

	st.dataframe(Lista)


	generar_grafica = st.checkbox('Generar gráfica')		

	if generar_grafica:
		#Generando la gráfica 
		st.subheader("Gráfico de barras de los datos procesados")
		grafica = plt.figure(figsize=(20,30))
		plt.ylabel('Item')
		plt.xlabel('Frecuencia')
		plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
		st.pyplot(grafica)

# --------------------------------------------------------------------------------
# FUNCIÓN PARA IMPLEMENTAR EL ALGORITMO
# --------------------------------------------------------------------------------

def aplicarApriori(data_file):
	#Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
	#level=0 especifica desde el primer índice
	#stack se encarga de quitar los nan para preparar los datos para aplicar el algoritmo.
	#De esta manera, no aparecen nan en la regla de asociación 
	data_file = data_file.stack().groupby(level=0).apply(list).tolist()

	#Obtenemos las reglas de configuración
	st.write("#### Soporte")
	st.write(info_soporte)
	soporte = st.number_input('Inserta el soporte: ',
								min_value=0.01)
	st.write("#### Confianza")
	st.write(info_confianza)
	confianza = st.number_input('Inserta la confianza: ',
								min_value=0.3)
	st.write("#### Elevación")
	st.write(info_elevacion)
	lift = st.number_input('Inserta la elevación (lift): ')

	reglas = apriori(data_file, 
                  	min_support=soporte, 
                   	min_confidence=confianza, 
                   	min_lift=lift)

	resultados = list(reglas)
	
	mostrar_resultados = st.checkbox('Mostrar resultados')		

	resultados_text = list()

	if mostrar_resultados:
		cont_regla = 0
		for item in resultados:
		  #El primer índice de la lista
		  emparejar = item[0]
		  cont_regla += 1
		  items = [x for x in emparejar]
		  regla = ("#### Regla " + str(cont_regla) + ": " +str((", ".join(item[0]))))
		  st.write(regla)

		  #El segundo índice de la lista
		  soporte_text=("Soporte: " + str(item[1]))
		  st.write(soporte_text)
		  #El tercer índice de la lista
		  confianza_text=("Confianza: " + str(item[2][0][2]))
		  st.write(confianza_text)

		  lift_text=("Lift: " + str(item[2][0][3]))
		  st.write(lift_text)

		  resultados_text.append(regla + " " + soporte_text + " " + confianza_text + " " + lift_text)
	
		#Convertir lista en texto
		raw_text = ' '.join(map(str, resultados_text))

		prueba = nfx.clean_text(raw_text)
		st.download_button(label='🔥 Descargar resultados 🔥', data = prueba, file_name = 'ReglasDeAsociacion.txt')


# --------------------------------------------------------------------------------
# FUNCIÓN PARA SUBIR ARCHIVOS 
# --------------------------------------------------------------------------------
#La siguiente función la utilizan todos los algoritmos, 
# ya que sirve para subir los archivos y mostrarlos
def set_archivoRA(): #Función para subir un archivo 
	data_file = st.file_uploader("Selecciona un archivo [.CSV]",
								type = ["csv"], key = 'ReglasA') #El usuario proporciona un archivo de extensión CSV
	if data_file is not None:
			data_file = pd.read_csv(data_file, header = None) 	 
			st.subheader('Visualización de los datos')    
			st.dataframe(data_file) #Muestra la data
			return data_file #Regresa el archivo subido 






