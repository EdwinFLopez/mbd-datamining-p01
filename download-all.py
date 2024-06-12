import os

import utils.download_data as dd
import utils.unzip_data as uz
from utils import npy2hdf5

if __name__ == '__main__':
    # Descargamos desde: https://zenodo.org:
    # * readme.txt (1K)
    # * metadata.csv (1.1M)
    # * audio_files.zip (993M)
    # * spectrograms.zip (2.8G)
    # #####################################

    # Creamos folder de intercambio de datos.
    if not os.path.exists("./data"):
        # Creating data folder, duh!
        os.makedirs("./data", exist_ok=True)

    # Descargamos los data files
    if not os.path.exists("./data/readme.txt"):
        dd.download_data("https://zenodo.org/records/7505820/files/readme.txt?download=1","./data/")
    if not os.path.exists("./data/metadata.csv"):
        dd.download_data("https://zenodo.org/records/7505820/files/metadata.csv?download=1","./data/")
    if not os.path.exists("./data/audio_files.zip"):
        dd.download_data("https://zenodo.org/records/7505820/files/audio_files.zip?download=1","./data/")
    if not os.path.exists("./data/spectrograms.zip"):
        dd.download_data("https://zenodo.org/records/7505820/files/spectrograms.zip?download=1","./data/")

    # Unzipping
    if os.path.exists("./data/audio_files.zip") and not os.path.exists("./data/audio_files"):
        uz.unzip_data("./data/audio_files.zip", "./data/audio_files")
    if os.path.exists("./data/spectrograms.zip") and not os.path.exists("./data/spectrograms"):
        uz.unzip_data("./data/spectrograms.zip", "./data/spectrograms")

    # Convertimos los espectrogramas en formato h5 para facilitar el procesamiento.
    if os.path.exists("./data/spectrograms") and not os.path.exists("./data/spects_h5"):
        npy2hdf5.convert_npy_to_hdf5(
            os.path.abspath("./data/spectrograms"),
            os.path.abspath("./data/spects_h5")
        )

    print("==============================================")
    print("Data folder created and ready to be processed.")
    print("==============================================")
