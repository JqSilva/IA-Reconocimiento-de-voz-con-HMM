import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


data_directory = './data/mini_speech_commands'

# ---------------------------------------
# Función para extraer características MFCC del audio
# ---------------------------------------
def extract_mfcc(path, nfilt=13, nfft=512):
    rate, sig = wav.read(path)
    mfcc_features = mfcc(sig, rate, numcep=nfilt, nfft=nfft)
    return mfcc_features

# ---------------------------------------
# Función para construir el conjunto de datos
# ---------------------------------------
def build_data_set(directory, nfilt=13, nfft=512, split=0.8):
    file_list = [f for f in os.listdir(directory) if os.path.splitext(f)[1] == '.wav']
    train_dataset = []
    test_dataset = []
    train_size = int(len(file_list) * split)
    
    # Dividir los datos en entrenamiento y prueba
    for file_name in file_list[:train_size]:
        path = os.path.join(directory, file_name)
        feature = extract_mfcc(path, nfilt, nfft)
        train_dataset.append(feature)
    
    for file_name in file_list[train_size:]:
        path = os.path.join(directory, file_name)
        feature = extract_mfcc(path, nfilt, nfft)
        test_dataset.append(feature)
    
    return train_dataset, test_dataset

# ---------------------------------------
# Función para entrenar modelos HMM
# ---------------------------------------
def entrenar_modelos_hmm(comandos, data_directory, nfilt=13, num_estados=5):
    modelos = {}
    for comando in comandos:
        # Construir los conjuntos de datos de entrenamiento
        train_data, _ = build_data_set(os.path.join(data_directory, comando), nfilt=nfilt)
        
        # Crear y entrenar el modelo HMM
        modelo = hmm.GaussianHMM(n_components=num_estados)
        modelo.fit(np.vstack(train_data))  # Unir los datos para entrenar el modelo
        modelos[comando] = modelo
    
    return modelos

# ---------------------------------------
# Lista de comandos
# ---------------------------------------
comandos = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Entrenar los modelos HMM
modelos_hmm = entrenar_modelos_hmm(comandos, data_directory)

# ---------------------------------------
# Función para reconocer un comando de audio
# ---------------------------------------
def reconocer_comando(archivo_audio, modelos, n_mfcc=13):
    caracteristicas = extract_mfcc(archivo_audio, nfilt=n_mfcc)
    probabilidades = {comando: modelo.score(caracteristicas) for comando, modelo in modelos.items()}
    return max(probabilidades, key=probabilidades.get)

# ---------------------------------------
# Ejemplo de reconocimiento
# ---------------------------------------
comando_reconocido = reconocer_comando('./data/mini_speech_commands/yes/0a7c2a8d_nohash_0.wav', modelos_hmm)
print("Comando reconocido:", comando_reconocido)

# ---------------------------------------
# Evaluación del sistema
# ---------------------------------------
y_true = []
y_pred = []

# Evaluar para cada conjunto de prueba
for comando, test_data in zip(comandos, [build_data_set(os.path.join(data_directory, c), nfilt=13)[1] for c in comandos]):
    for audio in test_data:
        y_true.append(comando)
        pred_comando = reconocer_comando(audio, modelos_hmm)
        y_pred.append(pred_comando)

# Mostrar la matriz de confusión y las métricas de clasificación
matriz_confusion = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:\n", matriz_confusion)
print("Informe de Clasificación:\n", classification_report(y_true, y_pred))
