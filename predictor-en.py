import os
import pandas as pd
import librosa as lb
import matplotlib.pyplot as plt
import utils.find_files as ff

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder


# ======================================================================================================================
def load_audio_data(filename: str, data_folder: str, reload: bool = False) -> pd.DataFrame:
    """Load audio data from a csv file and reload it if required."""

    # Load pre-created file.
    if os.path.exists(filename) and not reload:
        df = pd.read_csv(filename, sep=',', engine='python')
        df.reset_index(drop=True)
        return df;

    # Reload data from files
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
                    for feature in mfcc.T:
                        row = feature.tolist() + [bird_name]
                        data.append(row)

    columns = [f'mfcc_{i + 1}' for i in range(len(data[0]) - 1)] + ['label']
    df_feat = pd.DataFrame(data, columns=columns)
    if os.path.exists(filename):
        os.remove(filename)
    df_feat.to_csv(filename, index=False)  # Grabar matriz
    return df_feat


# ======================================================================================================================
if __name__ == "__main__":
    dataframe = load_audio_data(os.path.abspath("./data/birds_data_rf.csv"), os.path.abspath("./data/audio_files"))
    X = dataframe.drop(columns=['label'])
    Y = dataframe['label']

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Train a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_test = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy}")

    # Calculate the F1-score of the model
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    print(f"F1-score: {f1}")

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
