import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def find_files(root_dir):
    # sr_img(1284)_cmse(0.7237712144851685)_lpips(0.5037543177604675)_lpipskfs(0.6018092036247253)_lpipsmix(0.5312719345092773).png
    pattern = r'sr_img\((.*?)\)_cmse\((.*?)\)_lpips\((.*?)\)_lpipskfs\((.*?)\)_lpipsmix\((.*?)\)\.png'
    data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(dirpath)
        for filename in filenames:
            match = re.match(pattern, filename)
            if match:
                img_val, cmse_val, lpips_val, lpipskfs_val, lpipsmix_val = match.groups()
                file_path = os.path.join(dirpath, filename)
                dir_path = os.path.dirname(file_path)
                cmse_float = float(cmse_val)

                elements = dir_path.split(os.sep)
                loss_el = elements[-1]

                data.append({
                    'loss': loss_el,
                    'filename': filename,
                    'img': img_val,
                    'cmse': cmse_float,
                    'lpips': float(lpips_val),
                    'lpipskfs': float(lpipskfs_val),
                    'lpipsmix': float(lpipsmix_val)
                })
    df = pd.DataFrame(data, columns=['loss', 'filename', 'img', 'cmse', 'lpips', 'lpipskfs', 'lpipsmix'])
    df_grouped_by_path = df.drop('img', axis=1).drop('filename', axis=1).groupby('loss').agg(['mean', 'min', 'max', 'std'])
    df_grouped_by_file = df.drop('loss', axis=1).drop('filename', axis=1).groupby('img').agg(['mean', 'min', 'max', 'std'])

    by_loss = {}
    for loss in df['loss'].unique():
        by_loss[loss] = df[df['loss'] == loss].sort_values('img')

    from itertools import combinations
    # loss_combinations = list(combinations(by_loss.keys(), 2))
    loss_combinations = [(loss1, loss2) for loss1 in by_loss.keys() for loss2 in by_loss.keys()]

    # Tworzenie 3D ramki danych
    columns = ['loss1', 'loss2']
    for column in ['cmse', 'lpips', 'lpipskfs', 'lpipsmix']:
        columns.append(f'{column}_t_stat')
        columns.append(f'{column}_p_value')
        columns.append(f'{column}_wilcox_stat')
        columns.append(f'{column}_wilcox_p_value')

    df_3d = pd.DataFrame(columns=columns)

    # Wykonanie testu t-studenta dla każdej kombinacji wartości 'loss' i kolumn
    for loss1, loss2 in loss_combinations:
        row = {'loss1': loss1, 'loss2': loss2}
        for column in ['cmse', 'lpips', 'lpipskfs', 'lpipsmix']:
            first_seq = by_loss[loss1][column]
            second_seq = by_loss[loss2][column]
            t_stat, p_value = stats.ttest_ind(first_seq, second_seq)
            wilcox_stat, wilcox_p_value = stats.ranksums(first_seq, second_seq)
            row[f'{column}_t_stat'] = t_stat
            row[f'{column}_p_value'] = p_value
            row[f'{column}_wilcox_stat'] = wilcox_stat
            row[f'{column}_wilcox_p_value'] = wilcox_p_value
        df_3d = df_3d.append(row, ignore_index=True)

    return df, df_grouped_by_path, df_grouped_by_file, df_3d


with open("a:\\phd_artifacts\\_summary_test_files\\by_path.csv", "wt") as by_path:
    for the_folder in [
        "_probaRED_files",
        "_probaNIR_files",
        "_mus2_test_files",
        "_harvard_test_files",
    ]:
        # Example usage: find files in current directory and its subdirectories
        df, df_by_path, df_by_file, df_3d = find_files(f'a:\\phd_artifacts\\{the_folder}\\')
        # df.to_csv(f"a:\\phd_artifacts\\{the_folder}\\details.csv")
        # df_by_path.to_csv(f"a:\\phd_artifacts\\{the_folder}\\by_path.csv")
        #
        # by_path.write("======================\r\n")
        # by_path.write(f"{the_folder}\r\n")
        # by_path.write("======================\r\n")
        # df_by_path.to_csv(by_path)
        # df_by_file.to_csv(f"a:\\phd_artifacts\\{the_folder}\\by_file.csv")

        df_3d.to_csv(f"a:\\phd_artifacts\\stat_tests_{the_folder}.csv")


        print(df.head())
