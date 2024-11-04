import os
import joblib
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Función para extraer y normalizar características MFCC del audio de entrada
def extract_mfcc(path, num_features=13, nfft=512):
    rate, sig = wav.read(path)
    mfcc_features = mfcc(sig, rate, numcep=num_features, nfft=nfft)
    
    # Normalizar las características MFCC
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features)
    
    return mfcc_features

# Función para cargar los modelos desde una carpeta específica
def cargar_modelos(config_folder, comandos):
    modelos = {}
    for comando in comandos:
        model_path = os.path.join(config_folder, f'{comando}_model.pkl')
        if os.path.exists(model_path):
            modelos[comando] = joblib.load(model_path)
        else:
            print(f"Advertencia: No se encontró el modelo para '{comando}' en {config_folder}")
    return modelos

# Función para reconocer el comando basado en la probabilidad máxima
def reconocer_comando(archivo_audio, modelos, num_features=13):
    # Extraer características MFCC del archivo de audio
    caracteristicas = extract_mfcc(archivo_audio, num_features)
    
    # Calcular la probabilidad de las características para cada modelo
    probabilidades = {comando: modelo.score(caracteristicas) for comando, modelo in modelos.items()}
    
    # Seleccionar el comando con la mayor probabilidad
    comando_reconocido = max(probabilidades, key=probabilidades.get)
    return comando_reconocido

# Evaluar el sistema con el conjunto de prueba
def evaluar_modelos(data_directory, config_folder, comandos, num_features):
    modelos = cargar_modelos(config_folder, comandos)
    
    y_true = []
    y_pred = []

    for comando in comandos:
        # Ruta a la carpeta de prueba de cada comando
        test_path = os.path.join(data_directory, comando)
        file_list = [f for f in os.listdir(test_path) if os.path.splitext(f)[1] == '.wav']

        for file_name in file_list:
            archivo_audio = os.path.join(test_path, file_name)
            y_true.append(comando)
            y_pred.append(reconocer_comando(archivo_audio, modelos, num_features))

    # Calcular la matriz de confusión y las métricas
    matriz_confusion = confusion_matrix(y_true, y_pred, labels=comandos)
    print("\nMatriz de Confusión:\n", matriz_confusion)
    
    # Informe de clasificación que incluye precisión, sensibilidad y F1-score
    informe = classification_report(y_true, y_pred, labels=comandos)
    print("\nInforme de Clasificación:\n", informe)

# Lista de configuraciones para evaluar
configuraciones = [
    {"config_folder": './models/features_13_n_components_3_covtype_diag', "num_features": 13},
    {"config_folder": './models/features_13_n_components_7_covtype_full', "num_features": 13},
    {"config_folder": './models/features_20_n_components_5_covtype_diag', "num_features": 20},
    {"config_folder": './models/features_20_n_components_7_covtype_spherical', "num_features": 20}
]

# Definir comandos y ruta al directorio de datos
comandos = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']
data_directory = './data/mini_speech_commands_extracted/mini_speech_commands'

# Evaluar cada configuración
for config in configuraciones:
    print("\nEvaluando Configuración:")
    print(f"Carpeta: {config['config_folder']}")
    print(f"Num Features: {config['num_features']}")
    
    evaluar_modelos(data_directory, config["config_folder"], comandos, config["num_features"])


# --------------------------------------------------------------------
"""# Sección para probar tus propios audios
print("\nPrueba con tus propios audios:")

# Ruta a tus audios personales
mis_audios_directory = './data/mis_audios'  # Cambia esto a la ruta de tus audios
file_list = [f for f in os.listdir(mis_audios_directory) if os.path.splitext(f)[1] == '.wav']

# Usar una de las configuraciones (por ejemplo, la primera)
config = configuraciones[0]
modelos = cargar_modelos(config["config_folder"], comandos)
num_features = config["num_features"]

# Probar con tus audios
for file_name in file_list:
    archivo_audio = os.path.join(mis_audios_directory, file_name)
    comando_reconocido = reconocer_comando(archivo_audio, modelos, num_features)
    print(f"Comando reconocido para {file_name}: {comando_reconocido}")
    
    
data_directory = './data/mis_audios'  # Cambia esta ruta a donde están tus audios
# Modifica la llamada a evaluar_modelos para usar tu carpeta de audios
data_directory = './data/mis_audios'  # Ruta a tus audios organizados en subcarpetas
config = configuraciones[0]  # Elige la configuración que quieras usar

evaluar_modelos(data_directory, config["config_folder"], comandos, config["num_features"])"""