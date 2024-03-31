import os
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
def base_dcnn_algorithm():
    """Algoritmo base de dCNN para reconocimento de audios de pájaros"""
    # Set the path to the audio files folder
    audio_folder = "audio_files"

    # Initialize empty lists for spectrogram data and labels
    data = []
    labels = []

    # Iterate through the audio files in the folder
    for filename in os.listdir(audio_folder):
        if filename.endswith(".mp3"):
            # Load the audio file
            audio_path = os.path.join(audio_folder, filename)
            y, sr = lb.load(audio_path)

            # Extract spectrogram features
            spectrogram = np.abs(lb.stft(y))

            # Resize the spectrogram to a fixed shape (e.g., 128x128)
            spectrogram_resized = np.resize(spectrogram, (128, 128))

            # Append the spectrogram data and label to the respective lists
            data.append(spectrogram_resized)
            labels.append(filename.split(".")[0])  # Assuming the filename is in the format "bird_label.mp3"

    # Convert the lists to numpy arrays
    X = np.array(data)
    y = np.array(labels)

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Loads metadata file and removes overlapping entries for each spectrogram."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata no encontrado en {metadata_path}.")

    metadada_df = pd.read_csv(metadata_path, sep=';', header=0, engine='python')
    metadada_df.reset_index(drop=True)
    return metadada_df

# ======================================================================================================================
def load_audio_data(metadata: pd.DataFrame, filename: str, data_folder: str, reload: bool = False) -> pd.DataFrame:
    # Load pre-created file.
    if os.path.exists(filename) and not reload:
        df = pd.read_csv(filename, sep=',', engine='python')
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
                    #extract_features(data, bird_name, cut_audio, sr, window_length, hop_length)

    # create a dataframe with the features
    columns = ['zero_crossing_rate', 'spectral_centroid', 'spectral_flux', 'spectral_bandwidth', 'energy']
    columns.extend([f'mfcc_{i}' for i in range(13)] + ['label'])
    df_feat = pd.DataFrame(data, columns=columns)

    if os.path.exists(filename):
        os.remove(filename)

    df_feat.to_csv(filename, index=False)
    return df_feat


# ======================================================================================================================
if __name__ == '__main__':
    metadata_path = "./data/metadata.csv"

    print(f"loading metadata file: {metadata_path}")
    metadata = load_metadata(metadata_path)

# ======================================================================================================================
