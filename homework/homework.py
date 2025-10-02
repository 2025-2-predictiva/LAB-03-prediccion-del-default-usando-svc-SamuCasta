# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Importación de librerías necesarias para el proyecto
from sklearn.pipeline import Pipeline  # Para crear pipelines de transformaciones y modelos
from sklearn.compose import ColumnTransformer  # Para aplicar diferentes transformaciones a diferentes columnas
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler  # Para codificación categórica y escalado
from sklearn.feature_selection import SelectKBest, f_classif  # Para selección de características más relevantes
from sklearn.model_selection import GridSearchCV  # Para búsqueda de hiperparámetros con validación cruzada
from sklearn.decomposition import PCA  # Para reducción de dimensionalidad usando análisis de componentes principales
from sklearn.svm import SVC  # Para el modelo de máquina de vectores de soporte
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Para evaluación del modelo
import zipfile  # Para extraer archivos comprimidos
import pickle  # Para serializar y deserializar objetos Python
import gzip  # Para compresión de archivos
import json  # Para manejo de archivos JSON
import os  # Para operaciones del sistema operativo
import pandas as pd  # Para manipulación y análisis de datos

def limpiar_datos(datos):
    """
    Función para limpiar y preprocesar los datos del dataset.
    
    Args:
        datos (DataFrame): DataFrame con los datos sin procesar
        
    Returns:
        DataFrame: DataFrame limpio y procesado
    """
    # Crear una copia del DataFrame para evitar modificar el original
    datos = datos.copy()
    
    # Eliminar la columna 'ID' ya que no aporta información para la predicción
    datos = datos.drop('ID', axis=1)
    
    # Renombrar la columna objetivo para un nombre más simple y claro
    datos = datos.rename(columns={'default payment next month': 'default'})
    
    # Eliminar filas con valores faltantes (NaN) para asegurar datos completos
    datos = datos.dropna()
    
    # Filtrar registros con información válida: eliminar registros donde EDUCATION o MARRIAGE sean 0 (N/A)
    datos = datos[(datos['EDUCATION'] != 0 ) & (datos['MARRIAGE'] != 0)]
    
    # Agrupar niveles de educación superiores a 4 en la categoría 'others' (valor 4)
    # Esto simplifica las categorías y agrupa educación superior en una sola clase
    datos.loc[datos['EDUCATION'] > 4, 'EDUCATION'] = 4

    return datos

def modelo():
    """
    Función para crear el pipeline de machine learning con todas las transformaciones necesarias.
    
    Returns:
        Pipeline: Pipeline completo con preprocesamiento, PCA, selección de características y SVM
    """
    # Definir las variables categóricas que necesitan codificación one-hot
    categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']  
    
    # Definir las variables numéricas que necesitan escalado estándar
    numericas = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5","PAY_AMT6"
    ]

    # Crear un preprocesador que aplica diferentes transformaciones según el tipo de variable
    preprocesador = ColumnTransformer(
        transformers=[
            # Aplicar One-Hot Encoding a variables categóricas (convierte categorías en variables binarias)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas),
            # Aplicar escalado estándar a variables numéricas (media=0, desviación estándar=1)
            ('scaler', StandardScaler(), numericas)
        ],
        remainder='passthrough'  # Mantener otras columnas sin transformar
    )

    # Configurar selección de características usando puntuación F (ANOVA F-test)
    seleccionar_k_mejores = SelectKBest(score_func=f_classif)

    # Crear el pipeline completo con todos los pasos de procesamiento y modelado
    pipeline = Pipeline(steps=[
        # Paso 1: Preprocesamiento (one-hot encoding y escalado)
        ('preprocesador', preprocesador),
        # Paso 2: Reducción de dimensionalidad con PCA (análisis de componentes principales)
        ('pca', PCA()),
        # Paso 3: Selección de las K características más relevantes
        ("seleccionar_k_mejores", seleccionar_k_mejores),
        # Paso 4: Modelo de clasificación SVM con kernel radial
        ('clasificador', SVC(kernel='rbf', random_state=42))
    ])

    return pipeline

def hiperparametros(modelo, n_divisiones, x_entrenamiento, y_entrenamiento, puntuacion):
    """
    Función para optimizar los hiperparámetros del modelo usando validación cruzada.
    
    Args:
        modelo (Pipeline): Pipeline del modelo a optimizar
        n_divisiones (int): Número de divisiones para la validación cruzada
        x_entrenamiento (DataFrame): Características de entrenamiento
        y_entrenamiento (Series): Variable objetivo de entrenamiento
        puntuacion (str): Métrica de evaluación a usar
        
    Returns:
        GridSearchCV: Modelo optimizado con los mejores hiperparámetros
    """
    # Crear el objeto GridSearchCV para búsqueda exhaustiva de hiperparámetros
    estimador = GridSearchCV(
        estimator=modelo,  # El pipeline a optimizar
        # Definir la grilla de hiperparámetros a probar
        param_grid = {
            'pca__n_components': [20, 21],  # Número de componentes principales a conservar
            'seleccionar_k_mejores__k': [12],  # Número de características más relevantes a seleccionar
            'clasificador__kernel': ['rbf'],  # Tipo de kernel para SVM (radial basis function)
            'clasificador__gamma': [0.099]  # Parámetro gamma para el kernel RBF
        },
        cv=n_divisiones,  # Número de divisiones para validación cruzada
        refit=True,  # Reentrenar el modelo con los mejores parámetros en todo el dataset
        verbose=0,  # No mostrar detalles del proceso
        return_train_score=False,  # No retornar puntuaciones de entrenamiento
        scoring=puntuacion  # Métrica de evaluación (balanced_accuracy)
    )
    
    # Entrenar el modelo probando todas las combinaciones de hiperparámetros
    estimador.fit(x_entrenamiento, y_entrenamiento)

    return estimador

def metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Función para calcular las métricas de evaluación del modelo.
    
    Args:
        modelo: Modelo entrenado
        x_entrenamiento (DataFrame): Características de entrenamiento
        y_entrenamiento (Series): Variable objetivo de entrenamiento
        x_prueba (DataFrame): Características de prueba
        y_prueba (Series): Variable objetivo de prueba
        
    Returns:
        tuple: Tupla con las métricas de entrenamiento y prueba
    """
    # Realizar predicciones en el conjunto de entrenamiento
    y_pred_entrenamiento = modelo.predict(x_entrenamiento)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred_prueba = modelo.predict(x_prueba)

    # Calcular métricas para el conjunto de entrenamiento
    metricas_entrenamiento = {
        'type': 'metrics',  # Tipo de registro
        'dataset': 'train',  # Conjunto de datos
        # Precisión: proporción de predicciones positivas correctas
        'precision': (precision_score(y_entrenamiento, y_pred_entrenamiento, average='binary')),
        # Precisión balanceada: promedio de recall por clase (maneja desbalance)
        'balanced_accuracy':(balanced_accuracy_score(y_entrenamiento, y_pred_entrenamiento)),
        # Recall: proporción de casos positivos reales identificados correctamente
        'recall': (recall_score(y_entrenamiento, y_pred_entrenamiento, average='binary')),
        # F1-score: media armónica entre precisión y recall
        'f1_score': (f1_score(y_entrenamiento, y_pred_entrenamiento, average='binary'))
    }

    # Calcular métricas para el conjunto de prueba
    metricas_prueba = {
        'type': 'metrics',  # Tipo de registro
        'dataset': 'test',  # Conjunto de datos
        # Precisión: proporción de predicciones positivas correctas
        'precision': (precision_score(y_prueba, y_pred_prueba, average='binary')),
        # Precisión balanceada: promedio de recall por clase (maneja desbalance)
        'balanced_accuracy':(balanced_accuracy_score(y_prueba, y_pred_prueba)),
        # Recall: proporción de casos positivos reales identificados correctamente
        'recall': (recall_score(y_prueba, y_pred_prueba, average='binary')),
        # F1-score: media armónica entre precisión y recall
        'f1_score': (f1_score(y_prueba, y_pred_prueba, average='binary'))
    }

    return metricas_entrenamiento, metricas_prueba

def matriz(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Función para calcular las matrices de confusión del modelo.
    
    Args:
        modelo: Modelo entrenado
        x_entrenamiento (DataFrame): Características de entrenamiento
        y_entrenamiento (Series): Variable objetivo de entrenamiento
        x_prueba (DataFrame): Características de prueba
        y_prueba (Series): Variable objetivo de prueba
        
    Returns:
        tuple: Tupla con las matrices de confusión de entrenamiento y prueba
    """
    # Realizar predicciones en el conjunto de entrenamiento
    y_pred_entrenamiento = modelo.predict(x_entrenamiento)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred_prueba = modelo.predict(x_prueba)

    # Calcular matriz de confusión para el conjunto de entrenamiento
    cm_entrenamiento = confusion_matrix(y_entrenamiento, y_pred_entrenamiento)
    # Extraer los valores de la matriz: TN, FP, FN, TP
    # TN: Verdaderos negativos, FP: Falsos positivos, FN: Falsos negativos, TP: Verdaderos positivos
    tn_entrenamiento, fp_entrenamiento, fn_entrenamiento, tp_entrenamiento = cm_entrenamiento.ravel()

    # Calcular matriz de confusión para el conjunto de prueba
    cm_prueba = confusion_matrix(y_prueba, y_pred_prueba)
    # Extraer los valores de la matriz: TN, FP, FN, TP
    tn_prueba, fp_prueba, fn_prueba, tp_prueba = cm_prueba.ravel()

    # Estructurar la matriz de confusión del conjunto de entrenamiento
    matriz_entrenamiento = {
        'type': 'cm_matrix',  # Tipo de registro
        'dataset': 'train',  # Conjunto de datos
        # Casos donde el valor real es 0 (no default)
        'true_0': {
            'predicted_0': int(tn_entrenamiento),  # Predicción correcta: no default
            'predicted_1': int(fp_entrenamiento)   # Predicción incorrecta: falso positivo
        },
        # Casos donde el valor real es 1 (default)
        'true_1': {
            'predicted_0': int(fn_entrenamiento),  # Predicción incorrecta: falso negativo
            'predicted_1': int(tp_entrenamiento)   # Predicción correcta: verdadero positivo
        }
    }

    # Estructurar la matriz de confusión del conjunto de prueba
    matriz_prueba = {
        'type': 'cm_matrix',  # Tipo de registro
        'dataset': 'test',  # Conjunto de datos
        # Casos donde el valor real es 0 (no default)
        'true_0': {
            'predicted_0': int(tn_prueba),  # Predicción correcta: no default
            'predicted_1': int(fp_prueba)   # Predicción incorrecta: falso positivo
        },
        # Casos donde el valor real es 1 (default)
        'true_1': {
            'predicted_0': int(fn_prueba),  # Predicción incorrecta: falso negativo
            'predicted_1': int(tp_prueba)   # Predicción correcta: verdadero positivo
        }
    }

    return matriz_entrenamiento, matriz_prueba

def guardar_modelo(modelo):
    """
    Función para guardar el modelo entrenado en formato comprimido.
    
    Args:
        modelo: Modelo entrenado a guardar
    """
    # Crear el directorio 'files/models' si no existe
    os.makedirs('files/models', exist_ok=True)

    # Guardar el modelo en formato pickle comprimido con gzip
    # Esto reduce significativamente el tamaño del archivo
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)

def guardar_metricas(metricas):
    """
    Función para guardar las métricas de evaluación en formato JSON.
    
    Args:
        metricas (list): Lista de diccionarios con las métricas calculadas
    """
    # Crear el directorio 'files/output' si no existe
    os.makedirs('files/output', exist_ok=True)

    # Guardar las métricas en formato JSON Lines (cada línea es un objeto JSON)
    with open("files/output/metrics.json", "w") as f:
        for metrica in metricas:
            # Convertir cada diccionario de métrica a formato JSON
            json_line = json.dumps(metrica)
            # Escribir cada métrica en una línea separada
            f.write(json_line + "\n")


# =============================================================================
# EJECUCIÓN PRINCIPAL DEL PROGRAMA
# =============================================================================

# Paso 1: Cargar los datos desde archivos ZIP
# Cargar el conjunto de datos de prueba desde el archivo ZIP
with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as zip:
    with zip.open('test_default_of_credit_card_clients.csv') as f:
        datos_prueba = pd.read_csv(f)

# Cargar el conjunto de datos de entrenamiento desde el archivo ZIP
with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as zip:
    with zip.open('train_default_of_credit_card_clients.csv') as f:
        datos_entrenamiento = pd.read_csv(f)

# Paso 2: Limpiar y preprocesar los datos
# Aplicar la función de limpieza a ambos conjuntos de datos
datos_prueba = limpiar_datos(datos_prueba)
datos_entrenamiento = limpiar_datos(datos_entrenamiento)

# Paso 3: Dividir los datos en características (X) y variable objetivo (y)
# Separar las características de entrenamiento y la variable objetivo
x_entrenamiento, y_entrenamiento = datos_entrenamiento.drop('default', axis=1), datos_entrenamiento['default']
# Separar las características de prueba y la variable objetivo
x_prueba, y_prueba = datos_prueba.drop('default', axis=1), datos_prueba['default']

# Paso 4: Crear el pipeline del modelo
# Crear el pipeline completo con todas las transformaciones y el modelo
pipeline_modelo = modelo()

# Paso 5: Optimizar hiperparámetros con validación cruzada
# Usar GridSearchCV con 10 divisiones y balanced_accuracy como métrica
pipeline_modelo = hiperparametros(pipeline_modelo, 10, x_entrenamiento, y_entrenamiento, 'balanced_accuracy')

# Paso 6: Guardar el modelo optimizado
# Guardar el modelo entrenado en formato comprimido
guardar_modelo(pipeline_modelo)

# Paso 7: Evaluar el modelo - Calcular métricas de rendimiento
# Calcular precisión, recall, f1-score y balanced_accuracy para ambos conjuntos
metricas_entrenamiento, metricas_prueba = metricas(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

# Paso 8: Calcular matrices de confusión
# Calcular las matrices de confusión para evaluar el rendimiento detallado
matriz_entrenamiento, matriz_prueba = matriz(pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba)

# Paso 9: Guardar todas las métricas y matrices de confusión
# Guardar todos los resultados de evaluación en un solo archivo JSON
guardar_metricas([metricas_entrenamiento, metricas_prueba, matriz_entrenamiento, matriz_prueba])