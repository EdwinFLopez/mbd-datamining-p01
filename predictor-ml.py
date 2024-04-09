import numpy as np
import os
import librosa as lb
import matplotlib.pyplot as plt
import pandas as pd
import utils.find_files as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ======================================================================================================================
def extract_features(data: list, bird_name: str, cut_audio: np.array, sr: np.array, window_length: int,
                     hop_length: int) -> list:
    zcr = lb.feature.zero_crossing_rate(y=cut_audio, frame_length=window_length, hop_length=hop_length)
    spectral_centroid = lb.feature.spectral_centroid(y=cut_audio, sr=sr, n_fft=window_length, hop_length=hop_length)
    spectral_flux = lb.onset.onset_strength(
        S=lb.feature.melspectrogram(y=cut_audio, sr=sr, n_fft=window_length, hop_length=hop_length)
    )
    spectral_bandwidth = lb.feature.spectral_bandwidth(y=cut_audio, sr=sr, n_fft=window_length, hop_length=hop_length)
    energy = lb.feature.rms(y=cut_audio, frame_length=window_length, hop_length=hop_length)
    mfcc = lb.feature.mfcc(y=cut_audio, sr=sr, n_mfcc=13, n_fft=window_length, hop_length=hop_length)

    zcr = zcr.reshape(-1, 1)
    spectral_centroid = spectral_centroid.reshape(-1, 1)
    spectral_flux = spectral_flux.reshape(-1, 1)
    spectral_bandwidth = spectral_bandwidth.reshape(-1, 1)
    energy = energy.reshape(-1, 1)

    features = np.concatenate(
        (zcr, spectral_centroid, spectral_flux, spectral_bandwidth, energy, mfcc.T), axis=1
    )

    for feature in features:
        row = feature.tolist()
        row.append(bird_name)
        data.append(row)


# ======================================================================================================================
def load_features_dataframe(features_file: str, data_folder: str, reload: bool = False) -> pd.DataFrame:
    """Load features from a csv file and reload it if requested by the user"""

    # Load pre-created file.
    if os.path.exists(features_file) and not reload:
        df = pd.read_csv(features_file, sep=',', engine='python')
        df.reset_index(drop=True)
        return df;

    # Reload data from files
    files = ff.find_files(data_folder, ".mp3")
    data = []
    for bird_name in files.keys():
        for audio_path in files[bird_name]:
            text_path = audio_path.replace(".mp3", ".txt")
            # Verificamos que el audio tenga etiquetas
            if not os.path.exists(text_path):
                print(f"audio {audio_path} no tiene tags")
                continue

            # Cargamos el audio con librosa
            y, sr = lb.load(audio_path)
            with open(text_path, "r") as lines:
                for line in lines:
                    # Leer los marcadores del fragmento de audio
                    start_time, end_time, _ = line.strip().split("\t")

                    # Extraer el fragmento de audio
                    start = lb.time_to_samples(float(start_time))
                    end = lb.time_to_samples(float(end_time))
                    cut_audio = y[start:end]

                    # Verificar si el segmento está vacío y saltar al siguiente cuando verdadero.
                    if len(cut_audio) == 0:
                        print(f"audio {audio_path} con corte vacio [{start_time}:{end_time}]")
                        continue

                    # Extraer la ventana de características
                    window_length = len(cut_audio) // 4
                    hop_length = window_length // 4
                    extract_features(data, bird_name, cut_audio, sr, window_length, hop_length)

    # create a dataframe with the features
    columns = ['zero_crossing_rate', 'spectral_centroid', 'spectral_flux', 'spectral_bandwidth', 'energy']
    columns.extend([f'mfcc_{i}' for i in range(13)] + ['label'])
    df_feat = pd.DataFrame(data, columns=columns)

    if os.path.exists(features_file):
        os.remove(features_file)

    df_feat.to_csv(features_file, index=False)
    return df_feat


# ======================================================================================================================
if __name__ == "__main__":
    features_file = os.path.abspath("./data/birds_features.csv")
    audio_files = os.path.abspath("./data/audio_files")

    df_feat = load_features_dataframe(features_file, audio_files)

    # Training the Support Vector Machine (SVM)
    X = df_feat.drop(columns=['label'])
    y = df_feat['label']

    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Standardize the features because the SVM model is very sensitive to the scale of the variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train an SVM classifier with scaled features
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Evaluate the SVM model
    y_pred_test = svm_model.predict(X_test_scaled)

    # Accuracy of the SVM model
    accuracy_score = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy_score}")

    # Calculate the F1-score of the model
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    print("F1-score:", f1)

    # Additional evaluation metrics
    classification_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)
    print(f"classification report: \n{classification_report}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
    cmd.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Matriz de Confusion")
    plt.ylabel("Verdaderas")
    plt.xlabel("Predicciones")
    plt.show()
