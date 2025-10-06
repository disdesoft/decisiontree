# 📧 Clasificación de Correos SPAM con Árbol de Decisión

Este proyecto implementa un modelo de **aprendizaje supervisado** utilizando un **Árbol de Decisión** para clasificar correos electrónicos como **SPAM o NO-SPAM**, apoyado en las librerías de *scikit-learn* y una interfaz interactiva creada con **Streamlit**.

---

## 🚀 Objetivo

Evaluar el rendimiento de un modelo de **Decision Tree** aplicado a un conjunto de correos electrónicos, midiendo su **exactitud**, **F1-score** y **Z-score** a lo largo de múltiples ejecuciones.  
El propósito es analizar la **precisión y estabilidad** del clasificador frente a diferentes divisiones de los datos.

---

## 🧩 Estructura del proyecto

```bash
spam_decisiontree.py   # Script principal con Streamlit
emails__dataset.csv    # Dataset con los correos (Body, Sender, BodyLength, Spam)
README.md              # Documento actual
```

---

## 🧠 Tecnologías utilizadas

- **Python 3.10+**
- **scikit-learn** – Entrenamiento y evaluación del modelo
- **pandas / numpy** – Manipulación y análisis de datos
- **matplotlib** – Visualización de resultados
- **Streamlit** – Interfaz web interactiva
- **SciPy** – Cálculo de Z-score y manejo de matrices dispersas

---

## 📊 Descripción del dataset

El conjunto de datos contiene información de correos electrónicos con las siguientes columnas:

| Variable | Descripción |
|-----------|-------------|
| `Body` | Texto del mensaje del correo |
| `Sender` | Remitente del correo |
| `BodyLength` | Longitud del cuerpo del mensaje (opcional) |
| `Spam` | Variable objetivo (1 = SPAM, 0 = NO-SPAM) |

---

## ⚙️ Flujo del programa

### **1. Configuración general**
El usuario define los parámetros de ejecución desde la barra lateral de Streamlit:
- Número de ejecuciones (`num_runs`)
- Tamaño del conjunto de prueba (`test_size`)
- Profundidad máxima del árbol (`max_depth`)

### **2. Carga de datos**
Se lee el archivo `emails__dataset.csv` y se muestra una vista previa del contenido.

### **3. Preprocesamiento de variables**

- **Vectorización del texto:**  
  Se aplica `TfidfVectorizer(stop_words="english", max_features=500)` para transformar el texto del cuerpo de los correos en vectores numéricos.  
  Esto permite representar los mensajes según las palabras más relevantes, eliminando palabras comunes en inglés.

- **Codificación del remitente:**  
  Se usa `LabelEncoder` para transformar el nombre del remitente en números.  
  Este método se eligió por ser **más eficiente** que `OneHotEncoder`, ya que evita crear una matriz enorme con miles de columnas.  
  Además, el modelo solo necesita distinguir entre remitentes, no establecer relaciones entre ellos.

- **Incorporación de la longitud del cuerpo:**  
  Si la columna `BodyLength` está presente, se incluye como característica adicional.  
  Se utiliza `hstack` para **unir todas las matrices (texto, remitente y longitud)** en una sola matriz `X`.  
  Esto facilita el entrenamiento del modelo, ya que todas las variables quedan integradas en una sola estructura.

### **4. Entrenamiento y evaluación del modelo**

En cada ejecución:
1. Se dividen los datos en entrenamiento y prueba.  
2. Se entrena un modelo de **Árbol de Decisión** con la profundidad elegida.  
3. Se calculan las métricas de desempeño:
   - **Accuracy:** mide la proporción de aciertos.  
   - **F1-score:** combina precisión y recall, útil cuando las clases están desbalanceadas.  
   - **Z-score:** detecta ejecuciones con resultados atípicos o inconsistentes.

### **5. Resultados y estadísticas**

Se muestra una tabla con las métricas de cada ejecución y un resumen estadístico global (`describe()`), que incluye media, desviación estándar, valores mínimos y máximos.

### **6. Visualización**

Se genera un gráfico de líneas donde se observa cómo varía la **exactitud** a través de las ejecuciones, junto con una línea roja punteada que representa el promedio general.

### **7. Conclusiones automáticas**

El script calcula la **media y desviación estándar** de la exactitud y genera conclusiones automáticas sobre la estabilidad y el rendimiento del modelo.

---

## 📈 Resultados generales

- **Exactitud promedio:** ≈ 80%  
- **Desviación estándar:** muestra variaciones leves entre ejecuciones.  
- **F1-score:** refleja un buen balance entre precisión y recall.  
- **Z-score:** permite detectar ejecuciones con resultados atípicos.

El modelo demostró ser confiable y fácil de interpretar, aunque sensible a los cambios en la partición de los datos.

---

## 💬 Discusión

El **Árbol de Decisión** es un modelo sencillo y explicativo, ideal para comprender cómo se comportan los datos.  
Sin embargo, puede **sobreajustarse** si su profundidad no se controla adecuadamente.  
Para mejorar la precisión y estabilidad, podrían usarse modelos más avanzados como:

- **Random Forest:** genera muchos árboles de decisión y combina sus resultados para reducir el sobreajuste y mejorar la estabilidad.  
- **Gradient Boosting:** construye los árboles de forma secuencial, corrigiendo los errores de los anteriores. Es más preciso, aunque también más exigente en cómputo.

---

## 🧩 Conclusiones

- El modelo logró una precisión media del **80%**, lo que se considera aceptable para un conjunto de datos de texto sin procesamiento lingüístico avanzado.  
- Se observó **ligera variabilidad**, por lo que es recomendable ejecutar el modelo varias veces para validar su estabilidad.  
- Este proyecto sirve como **base sólida** para implementar técnicas más robustas de clasificación de texto, como Random Forest o Gradient Boosting.

---

## 🌐 Despliegue y código fuente

- **Repositorio:** [https://github.com/disdesoft/decisiontree](https://github.com/disdesoft/decisiontree)  
- **Aplicación en línea (Streamlit):** [https://arboldedecision.streamlit.app/](https://arboldedecision.streamlit.app/)
