import os
import pandas as pd
from scipy.io import loadmat

data_dir = os.path.join(os.path.dirname(__file__), 'data')
raw_data = loadmat(os.path.join(data_dir, 'Patient2_Event_Row_4_Data_zero_order_interpolation_chunk0.mat'))

start = 0
end = 14000

time = raw_data['Time'].squeeze()[start:end]
relative_time = raw_data['Relative Time (sec)'].squeeze()[start:end]
ecg_1 = raw_data['GE_WAVE_ECG_1_ID'].squeeze()[start:end]
ecg_2 = raw_data['GE_WAVE_ECG_2_ID'].squeeze()[start:end]
ecg_3 = raw_data['GE_WAVE_ECG_3_ID'].squeeze()[start:end]
time_uniform = raw_data['Time_uniform'].squeeze()[start:end]

df = pd.DataFrame({
    'Time': time,
    'Relative Time (sec)': relative_time,
    'ECG_1': ecg_1,
    'ECG_2': ecg_2,
    'ECG_3': ecg_3,})

df.to_csv(os.path.join(data_dir, 'chunk.csv'), index=False)

print("Data saved to chunk.csv")