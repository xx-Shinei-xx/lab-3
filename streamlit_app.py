import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson, norm, chi2_contingency
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV en el mismo directorio
data1 = pd.read_csv('data1.csv')

# Verificar los nombres de las columnas de data1
st.write("Nombres de Columnas:", data1.columns)

# Identificar el nombre correcto de la columna para las mediciones
columna_mediciones = "decaimiento solo con el aire"  # Ajustar según el nombre real de la columna

# Extraer las mediciones del DataFrame
mediciones = data1[columna_mediciones]

# Definir la aplicación Streamlit
def main():
    st.title('Ajuste de Distribución y Prueba χ²')

    # Mostrar una porción de los datos cargados
    st.subheader('Datos de Ejemplo')
    st.write(data1.head())

    # Mostrar histograma de las mediciones
    st.subheader('Histograma de Mediciones')
    plt.hist(mediciones, bins=20, color='skyblue', edgecolor='black')
    st.pyplot()

    # Botones para elegir el tipo de distribución
    tipo_distribucion = st.radio("Seleccionar Distribución:", ('Poisson', 'Gaussiana'))

    if tipo_distribucion == 'Poisson':
        # Ajuste de distribución de Poisson
        lambda_poisson = st.slider('λ (parámetro Poisson):', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        valores_esperados = [poisson.pmf(k, lambda_poisson) * len(mediciones) for k in range(len(mediciones))]
        frecuencias_observadas, _ = np.histogram(mediciones, bins=len(valores_esperados))
        _, valor_p, _, _ = chi2_contingency([frecuencias_observadas, valores_esperados])

        st.subheader(f'Distribución de Poisson (λ={lambda_poisson})')
        st.write(f'Valor p de la prueba χ²: {valor_p:.4f}')

    elif tipo_distribucion == 'Gaussiana':
        # Ajuste de distribución Gaussiana
        media = st.slider('Media:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        desviacion_estandar = st.slider('Desviación Estándar:', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        valores_esperados = [norm.pdf(x, media, desviacion_estandar) * len(mediciones) for x in mediciones]
        frecuencias_observadas, _ = np.histogram(mediciones, bins=len(valores_esperados))
        _, valor_p, _, _ = chi2_contingency([frecuencias_observadas, valores_esperados])

        st.subheader(f'Distribución Gaussiana (Media={media}, Desv. Estándar={desviacion_estandar})')
        st.write(f'Valor p de la prueba χ²: {valor_p:.4f}')

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
    
