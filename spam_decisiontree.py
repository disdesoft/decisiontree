# archivo: spam_decisiontree.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# =========================
# 1. Configuración general
# =========================
st.title("Clasificación de correos SPAM con Árbol de Decisión")

st.write("Este experimento ejecuta un Árbol de Decisión múltiples veces "
         "para evaluar su desempeño en términos de Exactitud, F1-score y Z-score.")

# =========================
# Sidebar con opciones
# =========================
st.sidebar.header("Parámetros de ejecución")

num_runs = st.sidebar.slider("Número de ejecuciones", min_value=50, max_value=200, value=50, step=10)
test_size = st.sidebar.slider("Proporción de prueba (test_size)", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
max_depth = st.sidebar.selectbox("Profundidad máxima del árbol (max_depth)", options=[None, 5, 10, 20, 50])

# =========================
# 2. Cargar datos
# =========================
df = pd.read_csv("emails__dataset.csv")

st.subheader("Vista previa del dataset")
st.write(df.head())

# =========================
# 3. Variables y objetivo
# =========================
target_col = "Spam"
y = df[target_col]

# Vectorizamos el cuerpo del correo
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_body = vectorizer.fit_transform(df["Body"].astype(str))

# Codificamos el remitente
le_sender = LabelEncoder()
X_sender = le_sender.fit_transform(df["Sender"].astype(str))
X_sender = X_sender.reshape(-1, 1)

# Incluimos la longitud del cuerpo si existe
if "BodyLength" in df.columns:
    X_length = df["BodyLength"].values.reshape(-1, 1)
    X = hstack([X_body, X_sender, X_length])
else:
    X = hstack([X_body, X_sender])

# =========================
# 4. Ciclo de ejecuciones
# =========================
results = []
for i in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append({"Ejecución": i+1, "Exactitud": acc, "F1-Score": f1})

# =========================
# 5. Resultados
# =========================
results_df = pd.DataFrame(results)
results_df["Z-Score"] = zscore(results_df["Exactitud"])

st.subheader("Resultados de las ejecuciones")
st.dataframe(results_df)

# =========================
# 6. Estadísticas
# =========================
st.subheader("Estadísticas globales")
st.write(results_df.describe())

# =========================
# 7. Gráfico
# =========================
st.subheader("Variación de la Exactitud en ejecuciones")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results_df["Ejecución"], results_df["Exactitud"], marker="o", label="Exactitud")
ax.axhline(results_df["Exactitud"].mean(), color="red", linestyle="--", label="Promedio")
ax.set_xlabel("Ejecución")
ax.set_ylabel("Exactitud")
ax.set_title("Exactitud del Árbol de Decisión en múltiples ejecuciones")
ax.legend()
st.pyplot(fig)

# =========================
# 8. Conclusiones
# =========================
st.subheader("Conclusiones")
mean_acc = results_df["Exactitud"].mean()
std_acc = results_df["Exactitud"].std()

conclusiones = []
conclusiones.append(f"La exactitud promedio fue de **{mean_acc:.4f}** con una desviación estándar de **{std_acc:.4f}**.")

if std_acc > 0.05:
    conclusiones.append("Existe una variación considerable entre ejecuciones; el modelo es sensible a los datos de entrenamiento.")
else:
    conclusiones.append("El modelo presenta resultados estables entre ejecuciones.")

if mean_acc < 0.8:
    conclusiones.append("La exactitud promedio es baja, se recomienda probar ajustes en el árbol de decisión o usar otras técnicas.")
else:
    conclusiones.append("La exactitud promedio es aceptable para este tipo de datos.")

st.write("\n".join(conclusiones))
