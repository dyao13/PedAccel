import os
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd

def load_mar_data(data_dir, pat_num):
    """
    Loads MAR data from a .csv file and returns it as a pandas DataFrame.

    Args:
        data_dir (str): Path to the directory containing the .mat file.
        pat_num (int): Patient number.

    Returns:
        pd.DataFrame: DataFrame containing MAR data.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    
    file_path = os.path.join(data_dir, f'Patient{pat_num}_SickBayMARData.csv')

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        raw_data = load_mat_file(os.path.join(data_dir, f'Patient{pat_num}_SickBayMARData.mat'))
        df = dict_to_df(raw_data, file_path)
        df = df.drop(columns=['description'])

    return df

def load_mat_file(file_path):
    """
    Loads .mat file and returns a dictionary.

    Args:
        file_path (str): Path to .mat file.

    Raises:
        FileNotFoundError: If the file does not exist.    

    Returns:
        dict: Dictionary containing data from .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    raw_data = loadmat(file_path)

    for key in list(raw_data.keys()):
        if key.startswith("__"):
            del raw_data[key]

    return raw_data

def dict_to_df(raw_data, save_path = None):
    """
    Converts dictionary to pandas DataFrame and (optionally) saves it as a .csv file.

    Args:
        raw_data (dict): Dictionary containing data.
        save_path (str, optional): Path to save .csv file. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing data.
    """
    for key in raw_data.keys():
        try:
            raw_data[key] = raw_data[key].squeeze()
        except:
            pass

    df = pd.DataFrame(raw_data)

    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return df

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    pat_nums = [2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15]

    for pat_num in tqdm(pat_nums):
        df = load_mar_data(data_dir, pat_num)

        print(df.keys())
        print(df.shape)
        print(df.head())

if __name__ == "__main__":
    main()