import os
import utils.download_data as dd
import utils.unzip_data as uz

if __name__ == '__main__':
    # Check if user wants to download the whole enchilada
    prompt = """
This will download the following files from https://zenodo.org: 
* readme.txt (1K)
* metadata.csv (1.1M)
* audio_files.zip (993M)
* spectrograms.zip (2.8G)
It's a big download, do you want to continue? (y/n)?
> """
    r = input(prompt).lower()
    if r not in ['y', 'yes']:
        print('Okay, see you later alligator.')
        exit(0)

    # User accepted to download.
    print("Understood. Downloading...")
    if not os.path.exists("./data"):
        # Creating data folder, duh!
        os.makedirs("./data", exist_ok=True)
    # Downloading data files
    if not os.path.exists("./data/readme.txt"):
        dd.download_data("https://zenodo.org/records/7505820/files/readme.txt?download=1","./data/")
    if not os.path.exists("./data/metadata.csv"):
        dd.download_data("https://zenodo.org/records/7505820/files/metadata.csv?download=1","./data/")
    if not os.path.exists("./data/audio_files.zip"):
        dd.download_data("https://zenodo.org/records/7505820/files/audio_files.zip?download=1","./data/")
    if not os.path.exists("./data/spectrograms.zip"):
        dd.download_data("https://zenodo.org/records/7505820/files/spectrograms.zip?download=1","./data/")

    # Unzipping the files if downloaded and not extracted
    if os.path.exists("./data/audio_files.zip") and not os.path.exists("./data/audio_files"):
        uz.unzip_data("./data/audio_files.zip", "./data/audio_files")
    if os.path.exists("./data/spectrograms.zip") and not os.path.exists("./data/spectrograms"):
        uz.unzip_data("./data/spectrograms.zip", "./data/spectrograms")
    print("==============================================")
    print("Data folder created and ready to be processed.")
    print("==============================================")
