import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson, norm, chi2_contingency
import matplotlib.pyplot as plt

# Cargar datos desde el archivo CSV en el mismo directorio
data1 = pd.read_csv('data1.csv')

# Identificar el nombre correcto de la columna para las mediciones
columna_mediciones = "decaimiento solo con el aire"  # Ajustar según el nombre real de la columna

# Extraer las mediciones del DataFrame
mediciones = data1[columna_mediciones]

# Definir la aplicación Streamlit
def main():
    st.title('Ajuste de Distribución y Prueba χ²')

    # Mostrar histograma de las mediciones con parámetros interactivos
    st.subheader('Histograma de Mediciones')
    
    # Parámetros interactivos del histograma
    num_bins = st.slider('Número de Bins:', min_value=5, max_value=50, value=20)
    range_min = st.slider('Valor Mínimo:', min_value=int(mediciones.min()), max_value=int(mediciones.max()), value=int(mediciones.min()))
    range_max = st.slider('Valor Máximo:', min_value=int(mediciones.min()), max_value=int(mediciones.max()), value=int(mediciones.max()))

    # Crear el histograma con los parámetros interactivos
    plt.hist(mediciones, bins=num_bins, range=(range_min, range_max), color='skyblue', edgecolor='black')
    plt.xlabel('Mediciones')
    plt.ylabel('Frecuencia')
    plt.title('Histograma Personalizable')
    st.pyplot()

    # Botones para elegir el tipo de distribución
    tipo_distribucion = st.radio("Seleccionar Distribución:", ('Poisson', 'Gaussiana'))

    if tipo_distribucion == 'Poisson':
        # Ajuste de distribución de Poisson
        lambda_poisson = st.slider('λ (parámetro Poisson):', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        valores_esperados = [poisson.pmf(k, lambda_poisson) * len(mediciones) for k in range(len(mediciones))]
        frecuencias_observadas, _ = np.histogram(mediciones, bins=len(valores_esperados))

    elif tipo_distribucion == 'Gaussiana':
        # Ajuste de distribución Gaussiana
        media = st.slider('Media:', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        desviacion_estandar = st.slider('Desviación Estándar:', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        valores_esperados = [norm.pdf(x, media, desviacion_estandar) * len(mediciones) for x in mediciones]
        frecuencias_observadas, _ = np.histogram(mediciones, bins=len(valores_esperados))

    # Convertir a tabla de contingencia
    tabla_contingencia = np.array([frecuencias_observadas, valores_esperados])

    # Realizar la prueba χ² de contingencia
    _, valor_p, _, _ = chi2_contingency(tabla_contingencia)

    # Mostrar resultado de la prueba χ²
    st.subheader('Resultado de la Prueba χ²')
    st.write(f'Valor p de la prueba χ²: {valor_p:.4f}')

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
