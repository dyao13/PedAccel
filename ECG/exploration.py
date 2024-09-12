from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')
raw_data = loadmat(os.path.join(data_dir, 'Patient2_Event_Row_4_Data_zero_order_interpolation_chunk0.mat'))

time = raw_data['Time'].squeeze()[:10000]
relative_time = raw_data['Relative Time (sec)'].squeeze()[:10000]
ecg_1 = raw_data['GE_WAVE_ECG_1_ID'].squeeze()[:10000]
ecg_2 = raw_data['GE_WAVE_ECG_2_ID'].squeeze()[:10000]
ecg_3 = raw_data['GE_WAVE_ECG_3_ID'].squeeze()[:10000]
time_uniform = raw_data['Time_uniform'].squeeze()[:10000]

df = pd.DataFrame({
    'Time': time,
    'Relative Time (sec)': relative_time,
    'ECG_1': ecg_1,
    'ECG_2': ecg_2,
    'ECG_3': ecg_3,})

original_sampling_rate = 250
target_sampling_rate = 125

nyquist_freq = target_sampling_rate / 2

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

cutoff_frequency = nyquist_freq
ecg_1_filtered = apply_lowpass_filter(ecg_1, cutoff_frequency, original_sampling_rate)
ecg_2_filtered = apply_lowpass_filter(ecg_2, cutoff_frequency, original_sampling_rate)
ecg_3_filtered = apply_lowpass_filter(ecg_3, cutoff_frequency, original_sampling_rate)

resample_factor = target_sampling_rate / original_sampling_rate

ecg_1_downsampled = resample(ecg_1_filtered, int(len(ecg_1_filtered) * resample_factor))
ecg_2_downsampled = resample(ecg_2_filtered, int(len(ecg_2_filtered) * resample_factor))
ecg_3_downsampled = resample(ecg_3_filtered, int(len(ecg_3_filtered) * resample_factor))

time_downsampled = resample(relative_time, len(ecg_1_downsampled))

df = pd.DataFrame({
    'Relative Time (sec)': time_downsampled,
    'ECG_1': ecg_1_downsampled,
    'ECG_2': ecg_2_downsampled,
    'ECG_3': ecg_3_downsampled,})

print(df.head())

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df['Relative Time (sec)'], df['ECG_1'], label="ECG_1 (Lead I)")
plt.title("ECG Signal 1 (Lead I)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df['Relative Time (sec)'], df['ECG_2'], label="ECG_2 (Lead II)")
plt.title("ECG Signal 2")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df['Relative Time (sec)'], df['ECG_3'], label="ECG_3 (Lead III)")
plt.title("ECG Signal 3 (Lead III)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()