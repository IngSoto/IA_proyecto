import streamlit as st
import pandas as pd  						# Para la manipulación y análisis de los datos

from IA_variables import *	
# --------------------------------------------------------------------------------
# FUNCIÓN PARA REGRESAR AL INICIO
# --------------------------------------------------------------------------------
def set_inicio():
	st.title('ALGORITHMIA')
	st.write("El siguiente video te mostrará el funcionamiento del entorno")
	#video1 = open("/Users/fatalista/Downloads/mid.mp4", "rb")
	st.video("https://youtu.be/YootvTxCOMQ")

	st.title('¿Qué es ALGORITHMIA?')
	st.write(info_algorithmia)

