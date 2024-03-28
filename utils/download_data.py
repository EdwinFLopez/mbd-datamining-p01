import os
import requests as rq


def download_data(url: str, target: str) -> None:
    """
    Download the given URL and save it to a target directory
    :param url: URL to get the data from
    :param target: Target directory where to save the data
    :return: None
    """
    try:
        response = rq.get(url)
        if response.ok:
            file_name = url.split('/')[-1]
            file_name = file_name.split('?')[0]
            file_path = os.path.join(target, file_name)
            print(f"Downloading {file_name}.....")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to {file_path}")
        else:
            print(f"Failed to download file from {url}")
    except Exception as e:
        print(f"An error occurred downloading {url}: {e}")
