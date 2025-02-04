import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

import preprocessing

def find_agitation(df):
    pattern = r'agitated|upset|fussy'
    mask = (df['Notes_TM'].str.contains(pattern, case=False, na=False))

    df_agitated = df[mask]
    df_not_agitated = df[~mask]

    return df_agitated, df_not_agitated

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        print('Data directory does not exist')
        return

    pat_nums = [4]

    for i in tqdm(pat_nums):
        # df_ecg1, df_ecg2, df_ecg3 = preprocessing.load_dfs(output_dir, data_dir, i)
        retro_df = preprocessing.read_retro_scores(data_dir, i)

        # df_agitated, df_not_agitated = find_agitation(retro_df)

        # agitated = np.array(df_agitated['TM SBS'])
        # not_agitated = np.array(df_not_agitated['TM SBS'])

        # _, p_val = stats.ttest_ind(agitated, not_agitated, equal_var=False)

        # print(f'p-value: {p_val}')

        raw_sickbay = preprocessing.load_mat(os.path.join(data_dir, f'Patient{i}_SICKBAY_10MIN_5MIN_Retro.mat'))

        for key in raw_sickbay.keys():
            if raw_sickbay[key].ndim == 2:
                new = np.empty(raw_sickbay[key].shape[0], dtype=object)

                for j in range(len(new)):
                    new[j] = raw_sickbay[key]
                
                raw_sickbay[key] = new

        raw_ecg = preprocessing.load_mat(os.path.join(data_dir, f'Patient{i}_10MIN_5MIN_ECG_SBSFinal.mat'))

        retro_df.insert(0, 'Time', preprocessing.format_times(retro_df['Time_uniform']))
        retro_df.drop(columns=['Time_uniform', 'Datetime'], inplace=True)

        raw_sickbay['start_time'] = preprocessing.format_times(raw_sickbay['start_time'])
        raw_sickbay['end_time'] = preprocessing.format_times(raw_sickbay['end_time'])

        raw_ecg['start_time'] = preprocessing.format_times(raw_ecg['start_time'])
        raw_ecg['end_time'] = preprocessing.format_times(raw_ecg['end_time'])

        sickbay_df = pd.DataFrame(raw_sickbay)
        ecg_df = pd.DataFrame(raw_ecg)

if __name__ == "__main__":
    main()
