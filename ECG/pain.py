import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

import preprocessing

def match_times(df1, df2):
    """
    Concatenates two dataframes based on matching times

    Args:
        df1 (pd.DataFrame): Dataframe with 'Time'
        df2 (pd.DataFrame): Dataframe with 'start_time' and 'end_time'

    Returns:
        pd.DataFrame: Concatenated dataframe with matching times
    """
    index1 = 0
    index2 = 0

    mask1 = np.zeros(len(df1), dtype=bool)
    mask2 = np.zeros(len(df2), dtype=bool)

    while index1 < len(df1) and index2 < len(df2):
        if df1['Time'][index1] < df2['start_time'][index2]:
            index1 += 1
        elif df1['Time'][index1] > df2['end_time'][index2]:
            index2 += 1
        else:
            mask1[index1] = True
            mask2[index2] = True

            index1 += 1
            index2 += 1

    df1 = df1[mask1].reset_index(drop=True)
    df2 = df2[mask2].reset_index(drop=True)

    return pd.concat([df1, df2], axis=1)

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

        # raw_sickbay['start_time'] = preprocessing.format_times(raw_sickbay['start_time'])
        # raw_sickbay['end_time'] = preprocessing.format_times(raw_sickbay['end_time'])

        raw_ecg['start_time'] = preprocessing.format_times(raw_ecg['start_time'])
        raw_ecg['end_time'] = preprocessing.format_times(raw_ecg['end_time'])

        # sickbay_df = pd.DataFrame(raw_sickbay)
        ecg_df = pd.DataFrame(raw_ecg)
        ecg_df.drop(columns=['sbs_score'], inplace=True)

        retro_df = retro_df[['Time', 'TM SBS']]
        match_df = match_times(retro_df, ecg_df)
        print(match_df)
        
if __name__ == "__main__":
    main()
