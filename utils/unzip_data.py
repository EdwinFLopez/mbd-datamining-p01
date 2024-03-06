import zipfile


def unzip_data(zip_file: str, target: str) -> None:
    """
    Utility to unzip a zip file and extract the data in a target folder.
    :param zip_file: file path of the zip file
    :param target: target folder where to extract the data
    :return: None
    """
    try:
        print(f"Unzipping {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(target)
        print(f"File unzipped and saved to {target}")
    except Exception as e:
        print(f"An error occurred: {e}")
