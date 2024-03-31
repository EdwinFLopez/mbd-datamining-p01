import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Set the path to the audio files folder
audio_folder = "audio_files"

# Initialize empty lists for spectrogram data and labels
spectrogram_data = []
labels = []

# Iterate through the audio files in the folder
for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):
        # Load the audio file
        audio_path = os.path.join(audio_folder, filename)
        y, sr = librosa.load(audio_path)

        # Extract spectrogram features
        spectrogram = np.abs(librosa.stft(y))

        # Resize the spectrogram to a fixed shape (e.g., 128x128)
        spectrogram_resized = np.resize(spectrogram, (128, 128))

        # Append the spectrogram data and label to the respective lists
        spectrogram_data.append(spectrogram_resized)
        labels.append(filename.split(".")[0])  # Assuming the filename is in the format "bird_label.mp3"

# Convert the lists to numpy arrays
spectrogram_data = np.array(spectrogram_data)
labels = np.array(labels)

# Encode the labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spectrogram_data, encoded_labels, test_size=0.2, random_state=42)

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
