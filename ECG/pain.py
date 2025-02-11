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

        # df_agitated, df_not_agitated = find_agitation(retro_df)

        # agitated = np.array(df_agitated['TM SBS'])
        # not_agitated = np.array(df_not_agitated['TM SBS'])

        # _, p_val = stats.ttest_ind(agitated, not_agitated, equal_var=False)

        # print(f'p-value: {p_val}')

        retro_df = preprocessing.load_retro(data_dir, i)
        ecg_df = preprocessing.load_ecg(data_dir, i)

        retro_df = retro_df[['Time', 'TM SBS']]
        match_df = preprocessing.match_times(retro_df, ecg_df)
        print(match_df)
        
if __name__ == "__main__":
    main()
