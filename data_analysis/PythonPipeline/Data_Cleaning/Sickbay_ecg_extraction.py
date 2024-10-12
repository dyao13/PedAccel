'''
Extracts ECG data from SickBay CSV files and saves them as .mat files
**This code is designed to run on the JHH SAFE Desktop Application**
'''

import pandas as pd
import numpy as np
from openpyxl import load_workbook
from scipy.io import savemat
import os

directory = os.chdir(r'S:\Sedation_monitoring\Sickbay_extract\Extract_250Hz')

def load_sickbay_mar():

    base_directory = r'S:\Sedation_monitoring\Sickbay_extract\Extract_250Hz'
    patient_info_add = 'Full_Patient_List_CCDA.xlsx'
    patient_info_excel = load_workbook(patient_info_add)
    patient_info = patient_info_excel['Sheet1']

    print('Patient List Loaded')

    for cell_a, cell_b in zip(patient_info['A'][1:], patient_info['B'][1:]):
        patient_num = cell_a.value
        patient_mrn = cell_b.value

        # Define dataframe and column names first...
        df_combined = pd.DataFrame(columns=['Relative Time (sec)', 'Time', 'GE_WAVE_ECG_1_ID', 'GE_WAVE_ECG_2_ID', 'GE_WAVE_ECG_3_ID'])
        
        patient_directory = os.path.join(base_directory, str(patient_mrn) + '_Study57_Tag123_EventList')
        # Check if the directory exists
        if os.path.exists(patient_directory):
            # Iterate through each CSV file in the patient's directory
            for file_name in os.listdir(patient_directory):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(patient_directory, file_name)
                    # Read the CSV file
                    chunk_iter = pd.read_csv(file_path, chunksize=1000000)
                    
                    chunk_counter = 0  # Initialize a counter for chunks
                    
                    for chunk in chunk_iter:
                        print("Chunking...")
                        # Select relevant columns
                        chunk = chunk[['Relative Time (sec)', 'Time', 'GE_WAVE_ECG_1_ID', 'GE_WAVE_ECG_2_ID', 'GE_WAVE_ECG_3_ID']]
                        
                        chunk['Time'] = pd.to_datetime(chunk['Time'])
                        chunk['Time_uniform'] = chunk['Time'].dt.strftime("%m/%d/%Y %I:%M:%S %p")
                        # Convert chunk to dictionary format for saving
                        data_dict = {col: chunk[col].values for col in chunk.columns}

                        # Save the chunk to a .mat file with a unique number
                        mat_file_name = f"Patient{patient_num}_{file_name.split('.')[0]}_chunk{chunk_counter}.mat"
                        savemat(mat_file_name, data_dict)
                        
                        print(f"Saved chunk {chunk_counter} of {file_name} to {mat_file_name}")
                        
                        # Increment the chunk counter
                        chunk_counter += 1
                        del chunk
        
# Finish one patient set extraction
        print(f"Saved {mat_file_name} for patient MRN: {patient_mrn}")
                  
load_sickbay_mar()