import os
import re

import numpy as np
import pandas as pd
from scipy import stats

import glob

def find_files(root_dir):
    results = []
    for file in glob.glob(os.path.join(root_dir, "details_other_8bit.csv")):
        df = pd.read_csv(file)
        ddd = os.path.basename(file)

        by_loss = {}
        for loss in df['model'].unique():
            by_loss[loss] = df[df['model'] == loss].sort_values('img')

        from itertools import combinations
        # loss_combinations = list(combinations(by_loss.keys(), 2))
        loss_combinations = [(loss1, loss2) for loss1 in by_loss.keys() for loss2 in by_loss.keys()]

        # Tworzenie 3D ramki danych
        columns = ['loss1', 'loss2']
        for column in ['PSNR','SSIM','LPIPS','LPIPS_KFS','LPIPS_MIX','cPSNR','cSSIM']:
            # columns.append(f'{column}_t_stat')
            columns.append(f'{column}_p_value')
            # columns.append(f'{column}_wilcox_stat')
            columns.append(f'{column}_wilcox_p_value')

        df_3d = pd.DataFrame(columns=columns)

        # Wykonanie testu t-studenta dla każdej kombinacji wartości 'loss' i kolumn
        for loss1, loss2 in loss_combinations:
            row = {'loss1': loss1, 'loss2': loss2}
            for column in ['PSNR','SSIM','LPIPS','LPIPS_KFS','LPIPS_MIX','cPSNR','cSSIM']:
                first_seq = by_loss[loss1][column]
                second_seq = by_loss[loss2][column]
                t_stat, p_value = stats.ttest_ind(first_seq, second_seq)
                wilcox_stat, wilcox_p_value = stats.ranksums(first_seq, second_seq)
                # row[f'{column}_t_stat'] = t_stat
                row[f'{column}_p_value'] = p_value
                # row[f'{column}_wilcox_stat'] = wilcox_stat
                row[f'{column}_wilcox_p_value'] = wilcox_p_value
            df_3d = df_3d.append(row, ignore_index=True)
        results.append((ddd, df_3d))
    return results


for the_folder in [
    # "__mus2_200px_results",
    "__mus2_results",
]:
    for id, df in find_files(f'a:\\phd_artifacts\\{the_folder}\\'):
        df.to_csv(f"a:\\phd_artifacts\\stat_tests_{the_folder}_20230615.csv")

