import numpy as np
import pandas as pd
from utils.File_saver import save_file, load_file
from sklearn.preprocessing import MinMaxScaler

# a function to load data from the Origin Excel
def load_data(file_path,sheet_names):
    # sheet name is a list of strings
    data = pd.read_excel(file_path, sheet_name=sheet_names)
    return data

# Define the function to standard scale the data
# The data is scaled to the range of [0, 1]
# The scaler should be formed using all the data
# The scaler is saved to the file
def standard_scale_data(data, train=True, scaler_file=None):
    if train:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        save_file(scaler_file, scaler)
    else:
        scaler = load_file(scaler_file)
        data = scaler.transform(data)
    return data

# Define the function to inverse the standard scaled data
def inverse_standard_scale_data(data, scaler_file):
    scaler = load_file(scaler_file)
    return scaler.inverse_transform(data)