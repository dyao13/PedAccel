import os
from scipy.io import loadmat
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class Preprocessing:
    """
    Preprocessing class for loading and processing MAR data.
    """

    @staticmethod
    def load_mar_data(data_dir, pat_num):
        """
        Loads MAR data from a .csv file and returns it as a pandas DataFrame.

        Parameters:
            data_dir (str): Path to the directory containing the .mat file.
            pat_num (int): Patient number. 

        Returns:
            pd.DataFrame: DataFrame containing MAR data.
        """    
        file_path = os.path.join(data_dir, f'Patient{pat_num}_SickBayMARData.csv')

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            raw_data = Preprocessing.load_mat_file(os.path.join(data_dir, f'Patient{pat_num}_SickBayMARData.mat'))
            df = Preprocessing.dict_to_df(raw_data, file_path)

        df = df.drop(columns=['description'])
        df.rename(columns={'mar_time': 'time'}, inplace=True)
        df = df[['time', 'dose', 'mar_action', 'med_name']]

        return df

    @staticmethod
    def load_mat_file(file_path):
        """
        Loads .mat file and returns a dictionary.

        Args:
            file_path (str): Path to .mat file.

        Parameters:
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

    @staticmethod
    def dict_to_df(raw_data, save_path = None):
        """
        Converts dictionary to pandas DataFrame and (optionally) saves it as a .csv file.

        Parameters:
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

    @staticmethod
    def match_times(df1, df2):
        """
        Matches times between two DataFrames.
        
        Parameters:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing matched times.
        """
        if df1.empty or df2.empty:
            return pd.DataFrame()

    def __match_times_time_time(df1, df2):
        pass

    def __match_times_time_startend(df1, df2):
        pass

    def __match_times_startend_startend(df1, df2):
        pass

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    pat_nums = [2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15]

    for pat_num in tqdm(pat_nums):
        df = Preprocessing.load_mar_data(data_dir, pat_num)

        print(df.keys())
        print(df.shape)
        print(df.head())

if __name__ == "__main__":
    main()