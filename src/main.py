import os
import re
import joblib
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from hmmlearn import hmm

# extraer y normalizar caracteristicas MFCC del audio de entrada
def extract_mfcc(path, num_features=13, nfft=1024):
    rate, sig = wav.read(path)
    mfcc_features = mfcc(sig, rate, numcep=num_features, nfft=nfft)
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features)
    return mfcc_features

# cargar los modelos desde una carpeta especifica
def cargar_modelos(config_folder, comandos):
    modelos = {}
    for comando in comandos:
        model_path = os.path.join(config_folder, f'{comando}_model.pkl')
        if os.path.exists(model_path):
            modelo = joblib.load(model_path)
            modelos[comando] = modelo
        else:
            print(f"Advertencia: No se encontró el modelo para '{comando}' en {config_folder}")
    return modelos

# reconocer el comando basado en  MFCC
def reconocer_comando_con_caracteristicas(caracteristicas, modelos):
    probabilidades = {comando: modelo.score(caracteristicas) for comando, modelo in modelos.items()}
    comando_reconocido = max(probabilidades, key=probabilidades.get)
    return comando_reconocido

#  evaluar el sistema con el conjunto de prueba desde test_data.pkl
def evaluar_modelos_con_test_data(test_data, config_folder, comandos, num_features):
    modelos = cargar_modelos(config_folder, comandos)
    y_true = []
    y_pred = []

    test_data_config = test_data.get(num_features)
    if not test_data_config:
        print(f"Advertencia: No hay conjunto de prueba para num_features = {num_features} en test_data.pkl. Omitiendo esta configuración.")
        return None

    for comando, caracteristicas_list in test_data_config.items():
        for caracteristicas in caracteristicas_list:
            if caracteristicas.shape[1] != num_features:
                print(f"Error: La cantidad de características ({caracteristicas.shape[1]}) no coincide con el modelo ({num_features}).")
                return None
            
            y_true.append(comando)
            y_pred.append(reconocer_comando_con_caracteristicas(caracteristicas, modelos))
    
    matriz_confusion = confusion_matrix(y_true, y_pred, labels=comandos)
    print("\nMatriz de Confusión:\n", matriz_confusion)
    informe = classification_report(y_true, y_pred, labels=comandos)
    print("\nInforme de Clasificación:\n", informe)
    f1_avg = f1_score(y_true, y_pred, average='weighted')
    return f1_avg

# evaluar el sistema con archivos de audio propios
def evaluar_modelos(data_directory, config_folder, comandos, num_features):
    modelos = cargar_modelos(config_folder, comandos)
    y_true = []
    y_pred = []
    
    for comando in comandos:
        test_path = os.path.join(data_directory, comando)
        file_list = [f for f in os.listdir(test_path) if os.path.splitext(f)[1] == '.wav']
        for file_name in file_list:
            archivo_audio = os.path.join(test_path, file_name)
            caracteristicas = extract_mfcc(archivo_audio, num_features)
            y_true.append(comando)
            y_pred.append(reconocer_comando_con_caracteristicas(caracteristicas, modelos))
    
    matriz_confusion = confusion_matrix(y_true, y_pred, labels=comandos)
    print("\nMatriz de Confusión:\n", matriz_confusion)
    informe = classification_report(y_true, y_pred, labels=comandos)
    print("\nInforme de Clasificación:\n", informe)
    f1_avg = f1_score(y_true, y_pred, average='weighted')
    return f1_avg

# detectar todas las configuraciones en el directorio "models"
def obtener_configuraciones():
    configuraciones = []
    model_dir = './models'
    pattern = r'n_components_(\d+)_dist_(\w+)_features_(\d+)'
    for folder_name in os.listdir(model_dir):
        match = re.match(pattern, folder_name)
        if match:
            n_components = int(match.group(1))
            distribution = match.group(2)
            num_features = int(match.group(3))
            config = {
                "config_folder": os.path.join(model_dir, folder_name),
                "n_components": n_components,
                "distribution": distribution,
                "num_features": num_features
            }
            configuraciones.append(config)
    return configuraciones

#  cargar y evaluar las configuraciones
def evaluacion(data_directory, opcion=1):
    configuraciones = obtener_configuraciones()
    mejores_resultados = []
    test_data = None

    if os.path.exists("test_data.pkl") and opcion == 2:
        with open("test_data.pkl", "rb") as f:
            test_data = joblib.load(f)
        print("\nConjunto de prueba cargado desde test_data.pkl")

    for config in configuraciones:
        print("\nEvaluando Configuración:")
        print(f"Carpeta: {config['config_folder']}")
        print(f"N Components: {config['n_components']}, Distribution: {config['distribution']}, Num Features: {config['num_features']}")
        
        if test_data:
            f1_avg = evaluar_modelos_con_test_data(test_data, config["config_folder"], comandos, config["num_features"])
            if f1_avg is None:
                continue
        else:
            f1_avg = evaluar_modelos(data_directory, config["config_folder"], comandos, config["num_features"])
        mejores_resultados.append((config['config_folder'], f1_avg))
        print(f"F1-Score promedio: {f1_avg:.4f}")

    mejor_config = max(mejores_resultados, key=lambda x: x[1])
    print(f"\nMejor configuración: {mejor_config[0]} con un F1-Score promedio de {mejor_config[1]:.4f}")

# Lista de comandos
comandos = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop']
data_directory = './data/Audios'

# Ejecutar la evaluacion
evaluacion(data_directory)