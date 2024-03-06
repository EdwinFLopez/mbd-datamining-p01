import os


def find_files(root_directory: str, extension: str) -> dict:
    """
    Find all files in a directory that have the specified extension, recursively.
    :param root_directory: starting directory
    :param extension: extension to search for
    :return: a dictionary with keys 'folder_name' and a list of file paths.
    """
    data = {}
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if root not in data.keys():
                    data[root] = []
                data[root].append(file_path)
        if 0 == len(data[root]):
            del data[root]
    return data
