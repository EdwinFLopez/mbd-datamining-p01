import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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
        else:
            print(f"{filename} no existe. Ignorando.")

    # Sobre-escribimos el dataframe con los archivos existentes
    metadata = pd.DataFrame(results)
    metadata.reset_index(drop=True, inplace=True)

    # Seleccionamos solo los espectrogramas que no se solapen para cada archivo
    results = []
    for audio_name in metadata['audio_name'].unique():
        segments = metadata[metadata['audio_name'] == audio_name]
        if len(segments) == 0:
            continue
        current = next(segments.itertuples())  # primero.
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
def load_spectrograms(metadata: pd.DataFrame, filename: str, spectrograms_folder: str,
                      reload: bool = False) -> pd.DataFrame:
    """
    Carga los data del archivo de metadata, removiendo los que no existen, y
    los que tengan segmentos solapados.
    """

    # Cargar archivo pre-generado
    if os.path.exists(filename) and not reload:
        df = pd.read_csv(filename, sep=',', engine='python')
        df.reset_index(drop=True)
        return df;

    data = []
    for spectrogram in metadata.itertuples():
        spectro_name = spectrogram.spectrogram_name
        species = spectrogram.species
        spectro_file_path = f"{spectrograms_folder}/{species}/{spectro_name}"
        if os.path.exists(spectro_file_path):
            npy = np.load(spectro_file_path)
            for entry in npy.T:
                row = entry.tolist() + [species]
                data.append(row)
        else:
            print(f"No existe el archivo {spectro_file_path}")

    columns = [f'col_{i + 1}' for i in range(len(data[0]) - 1)] + ['label']
    spectrograms_df = pd.DataFrame(data, columns=columns)

    # Recrear el archivo de data de los espectrogramas leídos.
    if os.path.exists(filename):
        os.remove(filename)

    spectrograms_df.to_csv(filename, index=False)
    return spectrograms_df


# ======================================================================================================================
def base_dcnn_algorithm(data: pd.DataFrame) -> None:
    """
    Algoritmo base de dCNN para reconocimento de audios de pájaros
    """
    x = data.drop(columns=['label'])
    y = data['label']

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
    class_report = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)
    print(f"classification report: \n{class_report}")

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
if __name__ == '__main__':
    cache_file = os.path.abspath("./data/birds_data_dcnn.csv")
    metadata_path = os.path.abspath("./data/metadata.csv")
    spectrogram_path = os.path.abspath("./data/spectrograms")

    print("Cargando metadata...")
    metadata = load_metadata(metadata_path, spectrogram_path)

    print("Cargando espectrogramas...")
    data = load_spectrograms(metadata, cache_file, spectrogram_path)

    print("Cargando modelo...")
    base_dcnn_algorithm(data)
    print("Terminamos base_dcnn_algorithm")
# ======================================================================================================================
