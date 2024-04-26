# Importa las bibliotecas necesarias
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

# Agrega una barra lateral para seleccionar la distribución
distribution_choice = st.sidebar.selectbox("Selecciona una distribución:", ["Gaussiana", "Poisson"])

# Genera datos de ejemplo según la distribución seleccionada
if distribution_choice == "Gaussiana":
    mu = st.slider("Media (μ)", min_value=0.0, max_value=10.0, value=5.0)
    sigma = st.slider("Desviación estándar (σ)", min_value=0.1, max_value=5.0, value=1.0)
    sample_size = st.slider("Tamaño de la muestra", min_value=10, max_value=100, value=50)
    sample_data = np.random.normal(mu, sigma, sample_size)
else:
    lambda_val = st.slider("Tasa (λ)", min_value=0.1, max_value=5.0, value=1.0)
    sample_size = st.slider("Tamaño de la muestra", min_value=10, max_value=100, value=50)
    sample_data = np.random.poisson(lambda_val, sample_size)

# Visualiza los datos generados
st.write("Datos generados:")
st.write(sample_data)

# Realiza la prueba de chi-cuadrado
observed_counts, _ = np.histogram(sample_data, bins="auto")
expected_counts = np.full_like(observed_counts, fill_value=sample_size / len(observed_counts))
chi2_statistic, p_value = stats.chisquare(observed_counts, expected_counts)

st.write(f"Estadístico de chi-cuadrado: {chi2_statistic:.4f}")
st.write(f"Valor p: {p_value:.4f}")

# Compara el valor p con un umbral (por ejemplo, 0.05) para determinar la significancia
alpha = 0.05
if p_value < alpha:
    st.write("Los datos difieren significativamente de la distribución esperada.")
else:
    st.write("No hay suficiente evidencia para concluir que los datos difieren de la distribución esperada.")

# Puedes agregar más visualizaciones y análisis según tus necesidades

# Ejecuta la aplicación con 'streamlit run nombre_del_archivo.py'
