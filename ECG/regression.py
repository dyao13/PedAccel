import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import root_mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    raw_data = loadmat(file_path)

    sbs_score = raw_data['sbs_score'].squeeze()
    start_time = raw_data['start_time'].squeeze()
    end_time = raw_data['end_time'].squeeze()
    ecg1 = raw_data['ecg1'].squeeze()
    ecg2 = raw_data['ecg2'].squeeze()
    ecg3 = raw_data['ecg3'].squeeze()

    for i in range(len(sbs_score)):
        start_time[i] = start_time[i].squeeze()
        end_time[i] = end_time[i].squeeze()

    for i in range(len(ecg1)):
        ecg1[i] = ecg1[i].squeeze()
        ecg2[i] = ecg2[i].squeeze()
        ecg3[i] = ecg3[i].squeeze()

    return sbs_score, start_time, end_time, ecg1, ecg2, ecg3

def extract_quality_signal(signal, sample_rate):
    quality_dict = {}
    step = int(sample_rate)
    window_size = int(10*sample_rate)

    for i in range(0, len(signal)-window_size, step):
        quality = nk.ecg_quality(signal[i:i+window_size],rpeaks = None, sampling_rate = sample_rate, method = 'zhao2018', approach = None)
        if(quality == 'Excellent' or quality == 'Barely acceptable'):
            quality_dict[i] = 1
        else:
            quality_dict[i] = 0
    
    quality_vals = list(quality_dict.values())
    index_tuple = get_continuous_indices(quality_vals)
    quality_keys = list(quality_dict.keys())

    return quality_keys[index_tuple[0]], quality_keys[index_tuple[1]]

def get_continuous_indices(arr):
    max_len = 0
    max_start = -1
    max_end = -1
    
    current_start = -1
    current_len = 0
    
    for i, value in enumerate(arr):
        if value == 1:
            if current_start == -1:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i - 1
            current_start = -1
            current_len = 0
    
    if current_len > max_len:
        max_len = current_len
        max_start = current_start
        max_end = len(arr) - 1
    
    return (max_start, max_end) if max_len > 0 else (-1, -1)

def get_metrics(sbs_score, ecg, fs):
    signals, _ = nk.ecg_peaks(ecg, sampling_rate=fs, correct_artifacts=True)
    df = nk.hrv(signals, sampling_rate=fs)
    df.insert(0, 'SBS_SCORE', sbs_score)

    return df

def save_metrics(file_path):
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    sbs_score, _, _, ecg1, ecg2, ecg3 = load_data(file_path)
    fs = 250
    # fs = int(1e9 * len(ecg1[0]) / (end_time[0] - start_time[0]))

    quality_ecg1 = np.empty(len(ecg1), dtype=object)
    quality_ecg2 = np.empty(len(ecg2), dtype=object)
    quality_ecg3 = np.empty(len(ecg3), dtype=object)

    for i in tqdm(range(len(sbs_score))):
        quality_start, quality_end = extract_quality_signal(ecg1[i], fs)
        quality_ecg1[i] = ecg1[i][quality_start:quality_end]

        quality_start, quality_end = extract_quality_signal(ecg2[i], fs)
        quality_ecg2[i] = ecg2[i][quality_start:quality_end]

        quality_start, quality_end = extract_quality_signal(ecg3[i], fs)
        quality_ecg3[i] = ecg3[i][quality_start:quality_end]
    
    df_ecg1 = pd.DataFrame()
    df_ecg2 = pd.DataFrame()
    df_ecg3 = pd.DataFrame()

    for i in tqdm(range(len(sbs_score))):
        metrics_1 = get_metrics(sbs_score[i], quality_ecg1[i], fs)
        metrics_2= get_metrics(sbs_score[i], quality_ecg2[i], fs)
        metrics_3 = get_metrics(sbs_score[i], quality_ecg3[i], fs)

        df_ecg1 = pd.concat([df_ecg1, metrics_1], ignore_index=True)
        df_ecg2 = pd.concat([df_ecg2, metrics_2], ignore_index=True)
        df_ecg3 = pd.concat([df_ecg3, metrics_3], ignore_index=True)
    
    df_ecg1.to_csv(os.path.join(output_dir, 'df_ecg1.csv'), index=False)
    df_ecg2.to_csv(os.path.join(output_dir, 'df_ecg2.csv'), index=False)
    df_ecg3.to_csv(os.path.join(output_dir, 'df_ecg3.csv'), index=False)

    return df_ecg1, df_ecg2, df_ecg3

def get_regression(df_ecg):
    df = pd.DataFrame({
        'SBS_SCORE': df_ecg['SBS_SCORE'],
        'HRV_MeanNN': df_ecg['HRV_MeanNN'],
        'HRV_SDNN': df_ecg['HRV_SDNN'],
        'HRV_LFHF': df_ecg['HRV_LFHF'],})

    if df.isna().any().any():
        print("hello")
        df.dropna(inplace=True)

    y = df['SBS_SCORE']
    X = df.drop(columns=['SBS_SCORE'])

    for key in X.keys():
        X[key] = StandardScaler().fit_transform(X[key].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print('root_mean_squared_error : ', root_mean_squared_error(y_test, predictions))
    print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
    print('score : ', model.score(X_test, y_test))

    return model

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_dir, 'Patient3_ECG.mat')

    if not os.path.exists(os.path.join(output_dir, 'df_ecg1.csv')) or not os.path.exists(os.path.join(output_dir, 'df_ecg2.csv')) or not os.path.exists(os.path.join(output_dir, 'df_ecg3.csv')):
        df_ecg1, df_ecg2, df_ecg3 = save_metrics(file_path)
    else:
        df_ecg1 = pd.read_csv(os.path.join(output_dir, 'df_ecg1.csv'))
        df_ecg2 = pd.read_csv(os.path.join(output_dir, 'df_ecg2.csv'))
        df_ecg3 = pd.read_csv(os.path.join(output_dir, 'df_ecg3.csv'))

    get_regression(df_ecg1)
    get_regression(df_ecg2)
    get_regression(df_ecg3)

if __name__ == '__main__':
    main()