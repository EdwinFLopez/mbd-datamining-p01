import os

import h5py
import numpy as np


def convert_npy_to_hdf5(input_folder, output_folder):
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        output_path = os.path.join(output_folder, item)

        if os.path.isdir(item_path):
            # If it's a subfolder, create a corresponding folder in the output directory
            os.makedirs(output_path, exist_ok=True)
            convert_npy_to_hdf5(item_path, output_path)
        elif item.endswith('.npy'):
            # If it's an .npy file, load and save it in the HDF5 file
            data = np.load(item_path)
            with h5py.File(os.path.join(output_folder, item.replace('.npy', '.h5')), 'w') as hf:
                hf.create_dataset(item.replace('.npy', ''), data=data)
