import streamlit as st
import pandas as pd
import kaggle.api.kaggle_api_extended 
import os
import zipfile
#from kaggle.api.kaggle_api_extended import KaggleApi
import toml

# Cargar credenciales de Kaggle
if 'KAGGLE_USERNAME' in st.secrets:
    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
else:
    config = toml.load("config.toml")
    os.environ['KAGGLE_USERNAME'] = config['kaggle']['username']
    os.environ['KAGGLE_KEY'] = config['kaggle']['key']


api = kaggle.api.kaggle_api_extended.KaggleApi()
api.authenticate()

# Función para descargar datasets de Kaggle
def download_dataset(dataset, file_name):
    api.dataset_download_file(dataset, file_name, path='datasets')
    zip_path = f'datasets/{file_name}.zip'
    csv_path = f'datasets/{file_name}'
    
    # Descomprimir el archivo si es necesario
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('datasets')
        os.remove(zip_path)
    
    return pd.read_csv(csv_path)

# Crear la aplicación Streamlit
st.title('Aplicación Interactiva de Ciencia de Datos')

# Sección 1: Cargar datasets desde Kaggle
st.header('1. Cargar Datasets desde Kaggle')

datasets = {
    'Titanic': ('heptapod/titanic', 'train.csv'),
    'Iris': ('uciml/iris', 'Iris.csv'),
    'Wine Quality': ('uciml/red-wine-quality-cortez-et-al-2009', 'winequality-red.csv'),
    'Diabetes': ('uciml/pima-indians-diabetes-database', 'diabetes.csv'),
    'Housing Prices': ('shree1992/housing-prices-dataset', 'Housing.csv'),
    'Heart Disease': ('ronitf/heart-disease-uci', 'heart.csv'),
    'Breast Cancer': ('uciml/breast-cancer-wisconsin-data', 'data.csv'),
    'MNIST': ('oddrationale/mnist-in-csv', 'mnist_train.csv'),
    'COVID-19': ('sudalairajkumar/novel-corona-virus-2019-dataset', 'covid_19_data.csv'),
    'World Happiness': ('unsdsn/world-happiness', '2019.csv')
}

dataset_name = st.selectbox('Seleccione un dataset', list(datasets.keys()))

if st.button('Cargar Dataset'):
    dataset_info = datasets[dataset_name]
    df = download_dataset(*dataset_info)
    st.write(f'Dataset {dataset_name} cargado con éxito!')
    st.write(df.head())

# Sección 2: Análisis Exploratorio de Datos (EDA)
st.header('2. Análisis Exploratorio de Datos (EDA)')
if 'df' in locals():
    st.write('Descripción del DataFrame:')
    st.write(df.describe())
    st.write('Información del DataFrame:')
    st.write(df.info())

# Sección 3: Visualización de Datos
st.header('3. Visualización de Datos')
if 'df' in locals():
    st.write('Gráficos interactivos aquí...')

# Sección 4: Modelado Predictivo
st.header('4. Modelado Predictivo')
if 'df' in locals():
    st.write('Modelos predictivos aquí...')
    
