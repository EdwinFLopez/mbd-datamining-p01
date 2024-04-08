import os
import pathlib

import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import utils.find_files as ff

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ======================================================================================================================
def base_dcnn_algorithm(data: list, labels: list):
    """Algoritmo base de dCNN para reconocimento de audios de pájaros"""

    # Set the path to the audio files folder
    spectograms_folder = os.path.abspath("./data/spectrograms/")
    # Initialize empty lists for spectrogram data and labels
    # data = []
    # labels = []
    # # Iterate through the audio files in the folder
    # for filename in os.listdir(spectrograms_folder):
    #     if filename.endswith(".mp3"):
    #         # Load the audio file
    #         audio_path = os.path.join(spectograms_folder, filename)
    #         y, sr = lb.load(audio_path)
    #
    #         # Extract spectrogram features
    #         spectrogram = np.abs(lb.stft(y))
    #
    #         # Resize the spectrogram to a fixed shape (e.g., 128x128)
    #         spectrogram_resized = np.resize(spectrogram, (128, 128))
    #
    #         # Append the spectrogram data and label to the respective lists
    #         data.append(spectrogram_resized)
    #         labels.append(filename.split(".")[0])  # Assuming the filename is in the format "bird_label.mp3"

    # Convert the lists to numpy arrays
    x = np.array(data)
    y = np.array(labels)

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.3, random_state=42)

    # Expand dimensions for CNN input
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode the labels
    num_classes = len(np.unique(encoded_labels))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build the CNN model using tf.keras
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Predict on the test set
    y_pred_test = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_test, axis=1)

    # Additional evaluation metrics
    classification_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)
    print(f"classification report: \n{classification_report}")

    # Calculate F1-score and Accuracy
    f1 = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)

    print("F1-score:", f1)
    print("Accuracy Score:", accuracy)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
    cmd.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Matriz de Confusion")
    plt.ylabel("Verdaderas")
    plt.xlabel("Predicciones")
    plt.show()


# ======================================================================================================================
def overlaps(record_a: tuple, record_b: tuple) -> bool:
    return record_a.sec_i < record_b.sec_i < record_a.sec_f \
        or record_a.sec_i < record_b.sec_f < record_b.sec_f


# ======================================================================================================================
def load_metadata(metadata_path: str, spectrogram_path: str) -> pd.DataFrame:
    """Loads metadata file and removes missing and overlapping candidates entries for each spectrogram."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata no encontrado en {metadata_path}.")

    if not os.path.exists(spectrogram_path):
        raise FileNotFoundError(f"Folder de espectrogramas no encontrado en {spectrogram_path}.")

    print(f"cargando metadata desde: {metadata_path}")
    print(f"validando espectrogramas desde: {spectrogram_path}")

    # Cargamos archivo de metadatos donde están los spectogramas
    metadata = pd.read_csv(metadata_path, sep=';', header=0, engine='python')

    # Filtramos los archivos que no han sido encontrados.
    results = []
    for record in metadata.iterrows():
        species = record[1].species
        spectrogram = record[1].spectrogram_name
        filename = f"{spectrogram_path}/{species}/{spectrogram}"
        if os.path.exists(filename):
            results.append(record[1])

    # Sobre-escribimos el dataframe con los archivos existentes
    metadata = pd.DataFrame(results)

    # Seleccionamos solo los spectrogramas que no se solapen para cada archivo
    results = []
    for audio_name in metadata['audio_name'].unique():
        segments = metadata[metadata['audio_name'] == audio_name]
        if len(segments) == 0:
            continue
        current = next(segments.itertuples())   # primero.
        results.append(current)
        for spectrogram in segments.itertuples():
            if not overlaps(current, spectrogram):
                results.append(spectrogram)
                current = spectrogram

    # Sobre-escribimos el dataframe con los archivos existentes con segmentos no solapados.
    metadata = pd.DataFrame(results)
    metadata.reset_index(drop=True)

    return metadata


# ======================================================================================================================
def load_spectrograms(metadata: pd.DataFrame, filename: str, data_folder: str, reload: bool = False) -> pd.DataFrame:

    # Cargar archivo pre-generado
    if os.path.exists(filename) and not reload:
        df = pd.read_csv(filename, sep=',', engine='python')
        df.reset_index(drop=True)
        return df;

    # Reload data from files
    files = ff.find_files(data_folder, ".npy")
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

                    # Cargar spectrogramas para el segmento dado.
                    # TODO:

    df_feat = pd.DataFrame(data)

    if os.path.exists(filename):
        os.remove(filename)

    df_feat.to_csv(filename, index=False)
    return df_feat


# ======================================================================================================================
if __name__ == '__main__':
    metadata_path = os.path.abspath("./data/metadata.csv")
    spectrogram_path = os.path.abspath("./data/spectrograms")
    metadata = load_metadata(metadata_path, spectrogram_path)

    # TODO: Ajustar carga de espectrogramas
    # load_audio_data(metadata, os.path.abspath("./data/birds_data_dcnn.csv"), os.path.abspath("./data/audio_files"))

    # TODO: ajustar para cargar espectrogramas de training y labels obtenidos.
    # Este es el punto de ejecución del DCNN
    print("El método base_dcnn_algorithm() aun no se encuentra completo.")
# ======================================================================================================================
