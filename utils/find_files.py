import os


def find_files(root_directory: str, extension: str) -> dict:
    """
    Find all files in a directory that have the specified extension, recursively.
    :rtype: object
    :param root_directory: starting directory
    :param extension: extension to search for
    :return: a dictionary with keys 'folder_name' and a list of file paths.
    """
    data = {}
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.abspath(os.path.join(root, file))
                key = os.path.basename(root)
                if key not in data.keys():
                    data[key] = []
                data[key].append(file_path)

    # Remove empty lists (should not happen)
    for key in data.keys():
        if len(data[key]) == 0:
            print(f"WARN: No file in {key}")
            del data[key]
    return data
