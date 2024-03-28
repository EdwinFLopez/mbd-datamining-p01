import os.path

import utils.find_files as ff

if __name__ == '__main__':
    files = ff.find_files("./data/audio_files/", ".mp3")
    print([key for key in files.keys()])
