import streamlit as st
import pandas as pd
# Importar datos de otro sublime
from IA_functions import *
from IA_functionts_RA import *
from IA_functions_CJ import *
from IA_functions_RL import *
from IA_functions_AD import *

st.set_page_config(page_title='AlgorithmIA')

st.sidebar.header('BIENVENIDO A ALGORITHMIA')
st.sidebar.markdown('Por favor, selecciona algún algoritmo')

# Pone el radio-button en horizontal. Afecta a todos los radio button de una página.
# Por eso está puesto en este que es general a todo
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

menu = st.sidebar.radio("",('Inicio',
						'Reglas de asociación',
						'Métricas de distancia', 
						'Clustering', 
						'Clasificación (R. Logística)', 
						'Árboles de Decisión (Pronóstico y Clasificación)'))


st.sidebar.markdown('---')
st.sidebar.write('Soto Hernández V. Ivan | Diciembre 2021 whyadie@gmail.com')

if menu == 'Inicio':
    set_inicio()
elif menu == 'Métricas de distancia':
   set_metricasDistancia()
elif menu == 'Reglas de asociación':
   set_reglasAsociacion()
elif menu == 'Clustering':
   set_clustering()
elif menu == 'Clasificación (R. Logística)':
   set_regresion()
elif menu == 'Árboles de Decisión (Pronóstico y Clasificación)':
   set_arboles()


























