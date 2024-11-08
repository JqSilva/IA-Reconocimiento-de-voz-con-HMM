import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from hmmlearn import hmm
import joblib
from sklearn.preprocessing import StandardScaler
import pickle  # Para guardar los datos en test_data.pkl
import shutil

# Función para extraer características MFCC del audio
def extract_mfcc(path, num_features=13, nfft=1024):
    rate, sig = wav.read(path)
    mfcc_features = mfcc(sig, rate, numcep=num_features, nfft=nfft)
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features)
    return mfcc_features

# Función para construir el conjunto de datos
def build_data_set(directory, num_features=13, nfft=1024, split=0.8):
    file_list = [f for f in os.listdir(directory) if os.path.splitext(f)[1] == '.wav']
    train_dataset = []
    test_dataset = []
    train_size = int(len(file_list) * split)
    
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
comandos = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop']
configuraciones = [
    {"num_features": 14, "n_components": 15, "distribution": "gmm"}
]

def entrenar_modelo():
    if os.path.exists('models'):
        shutil.rmtree('models')
    os.makedirs('models')
    
    for config in configuraciones:
        num_features = config["num_features"]
        n_components = config["n_components"]
        distribution = config["distribution"]

        config_folder = f'models/n_components_{n_components}_dist_{distribution}_features_{num_features}'
        os.makedirs(config_folder, exist_ok=True)

        print(f"\nEntrenando modelos con num_features={num_features}, n_components={n_components}, distribution={distribution}...")

        for comando in comandos:
            print(f"Entrenando el modelo para '{comando}'...")
            train_features, _ = build_data_set(os.path.join(data_directory, comando), num_features=num_features)
            
            if len(train_features) < n_components:
                print(f"Advertencia: Conjunto de datos insuficiente para '{comando}' con {n_components} componentes.")
                continue  # Omitir entrenamiento si los datos son insuficientes

            model_path = os.path.join(config_folder, f'{comando}_model.pkl')
            
            if distribution == "gaussian":
                model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
            elif distribution == "gmm":
                model = hmm.GMMHMM(n_components=n_components, covariance_type="diag", n_mix=4)
            else:
                print(f"Distribución '{distribution}' no reconocida. Omitiendo.")
                continue

            # Ajuste del modelo
            try:
                model.fit(np.vstack(train_features))
                joblib.dump(model, model_path)
                print(f"Modelo para '{comando}' guardado en {model_path}")
            except ValueError as e:
                print(f"Error al ajustar el modelo para '{comando}': {e}")

    print("Entrenamiento completado y modelos guardados en las subcarpetas correspondientes.")

def guardar_conjunto_prueba():
    test_data = {}

    # Generar los conjuntos de prueba separados por num_features y n_components
    for config in configuraciones:
        num_features = config["num_features"]
        n_components = config["n_components"]
        
        # Crear una entrada en el diccionario para cada configuración específica de num_features
        if num_features not in test_data:
            test_data[num_features] = {}

        for comando in comandos:
            # Obtener el conjunto de entrenamiento y el conjunto de prueba divididos en un 80/20
            _, test_features = build_data_set(os.path.join(data_directory, comando), num_features=num_features)
            test_data[num_features][comando] = test_features

    # Guardar test_data en test_data.pkl
    test_data_path = 'test_data.pkl'
    with open(test_data_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Conjunto de prueba guardado en {test_data_path}")

# Ejecutar solo la función de guardado de conjunto de prueba y entrenamiento si es necesario
if __name__ == "__main__":
    entrenar_modelo()
    guardar_conjunto_prueba()