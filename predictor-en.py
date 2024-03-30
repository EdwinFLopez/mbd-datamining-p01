# Adding a linear regression model for outlier detection and filtering before training 
# the random forest classifier by using the residuals obtained (residuals greater than 
# 2 are considered outliers and filtered out before training the random forest classifier).

import os
import numpy as np
import pandas as pd
import librosa as lb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


def load_audio_data(filename: str, data_folder: str, reload: bool = False) -> pd.DataFrame:
    """Load audio data from a csv file and reload it if required."""

    # Load pre-created file.
    if os.path.exists(filename) and not reload:
        df = pd.read_csv(filename, sep=',', engine='python')
        df.reset_index(drop=True)
        return df;

    # Reload data from files
    import utils.find_files as ff
    files = ff.find_files(data_folder, ".mp3")
    data = []
    labels = []
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

                    # Definir la ventana de características
                    window_length = len(cut_audio) // 4
                    hop_length = window_length // 4

                    mfcc = lb.feature.mfcc(y=cut_audio, sr=sr, n_fft=window_length, hop_length=hop_length)
                    mfcc_flat = mfcc.flatten() # Convertir a un array 1D

                    for feature in mfcc.T:
                        row = feature.tolist() + [bird_name]
                        data.append(row)

    columns = [f'mfcc_{i+1}' for i in range(len(data[0]) - 1)] + ['label']
    df_feat = pd.DataFrame(data, columns=columns)
    if os.path.exists(filename):
        os.remove(filename)
    df_feat.to_csv(filename, index=False) # Grabar matriz
    return df_feat


if __name__ == "__main__":
    dataframe = load_audio_data("./data/birds_data_rf.csv", "./data/audio_files")

    # Convert the lists to numpy arrays
    X = np.array(dataframe.drop(columns=['label']))
    Y = np.array(dataframe['label'])

    print(X.shape)
    print(Y.shape)

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Fit a linear regression model to detect outliers
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    # Define outliers as those with residuals greater than 2
    outliers = np.abs(regression_model.predict(X_train) - y_train) > 2

    # Remove outliers from the training data
    X_train_filtered = X_train[~outliers]
    y_train_filtered = y_train[~outliers]

    # Convert the filtered audio data to a feature matrix using CountVectorizer
    #vectorizer = CountVectorizer(lowercase=False, decode_error='ignore')
    #X_train_features = vectorizer.fit_transform(X_train_filtered) # Falla aquí. Hay que quitar la regresion
    #X_test_features = vectorizer.transform(X_test)

    # Train a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(X_train_filtered, y_train_filtered)

    # Make predictions on the test set
    y_pred_test = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Accuracy: ", accuracy)

    # Calculate the F1-score of the model
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    print("F1-score:", f1)

    # Additional evaluation metrics
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Assuming 'class_names' is a list containing your class labels
    class_names = ["Acrocephalus arundinaceus", "Acrocephalus melanopogon", "Acrocephalus scirpaceus",
                   "Alcedo atthis", "Anas platyrhynchos", "Anas strepera", "Ardea purpurea",
                   "Botaurus stellaris", "Charadrius alexandrinus", "Ciconia ciconia",
                   "Circus aeruginosus", "Coracias garrulus", "Dendrocopos minor",
                   "Fulica atra", "Gallinula chloropus", "Himantopus himantopus",
                   "Ixobrychus minutus", "Motacilla flava", "Porphyrio porphyrio", "Tachybaptus ruficollis"]

    # Create a DataFrame for better visualization (optional, but recommended)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
