import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from hmmlearn import hmm
import joblib
from sklearn.preprocessing import StandardScaler
import shutil

# Crear la carpeta 'models' si no existe
if os.path.exists('models'):
    shutil.rmtree('models')  # Eliminar toda la carpeta 'models' y su contenido
os.makedirs('models')  # Crear la carpeta 'models' de nuevo

# Función para extraer características MFCC del audio
def extract_mfcc(path, num_features=13, nfft=512):
    rate, sig = wav.read(path)
    mfcc_features = mfcc(sig, rate, numcep=num_features, nfft=nfft)
    
    # Normalizar las características MFCC
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features)
    return mfcc_features

# Función para construir el conjunto de datos
def build_data_set(directory, num_features=13, nfft=512, split=0.8):
    file_list = [f for f in os.listdir(directory) if os.path.splitext(f)[1] == '.wav']
    train_dataset = []
    test_dataset = []
    train_size = int(len(file_list) * split)
    
    # Dividir los datos en entrenamiento y prueba
    for file_name in file_list[:train_size]:
        path = os.path.join(directory, file_name)
        feature = extract_mfcc(path, num_features, nfft)
        train_dataset.append(feature)
    
    for file_name in file_list[train_size:]:
        path = os.path.join(directory, file_name)
        feature = extract_mfcc(path, num_features, nfft)
        test_dataset.append(feature)
    
    return train_dataset, test_dataset

# Definir la ruta al directorio de datos
data_directory = './data/mini_speech_commands_extracted/mini_speech_commands'

# Lista de comandos para entrenar
comandos = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go']

# Configuraciones para los modelos HMM
configuraciones = [
    {"num_features": 13, "n_components": 3, "covariance_type": "diag"},
    {"num_features": 20, "n_components": 5, "covariance_type": "diag"},
    {"num_features": 13, "n_components": 7, "covariance_type": "full"},
    {"num_features": 20, "n_components": 7, "covariance_type": "spherical"},
]

# Entrenar modelos HMM para cada configuración
for config in configuraciones:
    num_features = config["num_features"]
    n_components = config["n_components"]
    covariance_type = config["covariance_type"]

    # Crear una subcarpeta para la configuración actual
    config_folder = f'models/features_{num_features}_n_components_{n_components}_covtype_{covariance_type}'
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)

    print(f"\nEntrenando modelos con num_features={num_features}, n_components={n_components} y covariance_type={covariance_type}...")

    # Entrenar modelos para cada comando
    for comando in comandos:
        print(f"Entrenando el modelo para '{comando}'...")
        # Construir los conjuntos de datos
        train_features, _ = build_data_set(os.path.join(data_directory, comando), num_features=num_features)
        
        # Crear y entrenar el modelo HMM Gaussiano
        model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type)
        model.fit(np.vstack(train_features))
        
        # Guardar el modelo entrenado
        model_path = os.path.join(config_folder, f'{comando}_model.pkl')
        joblib.dump(model, model_path)

print("Entrenamiento completado y modelos guardados en las subcarpetas correspondientes.")
