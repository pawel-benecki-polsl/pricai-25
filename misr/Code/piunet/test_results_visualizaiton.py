import os
import re

import numpy as np
import pandas as pd
import pandas as pd

the_folder = "_mus2_test_files"
# the_folder = "_probaNIR_test_files"
# the_folder = "_probaRED_test_files"
# the_folder = "_harvard_test_files"
# Example usage: find files in current directory and its subdirectories
details = pd.read_csv(f"a:\\phd_artifacts\\{the_folder}\\details.csv")
col = 'lpips_kfsDIVlpips'
details[col] = details['lpipskfs'] / details['lpips']
details = details.sort_values(col)
largest = details.tail(50)
smallest = details.head(50)

def printout(filtered):
    # for img_number, vall in zip(img_numbers, vals):
    #     filtered = details[details['img'] == img_number] and details[details[col] == vall]
        print("<table>")
        print("<tr><td>img</td><td>hr</td><td>lr</td><td>descr</td>")
        for idx, row in filtered.iterrows():
            print("<tr>")
            path = f"a:\\phd_artifacts\\{the_folder}\\{row['loss']}"
            path2 = f"a:\\phd_artifacts\\{the_folder}\\{row['loss']}"
            path3 = f"a:\\phd_artifacts\\{the_folder}\\{row['loss']}"
            print(f"<td><img src='{path}\\{row['filename']}' /></td>")
            # print(f"<td><img src='{path2}\\{row['filename']}' /></td>")
            # print(f"<td><img src='{path3}\\{row['filename']}' /></td>")
            print(f"<td><img src='a:\\phd_artifacts\\{the_folder}\\MuS2_CPSNR\\hr_img({str(row['img']).zfill(4)}).png' /></td>")
            print(f"<td><img src='a:\\phd_artifacts\\{the_folder}\\MuS2_CPSNR\\lr0_img({str(row['img']).zfill(4)}).png' /></td>")
            print(f"<td>IDX: {idx}: {path}<br />cMSE: {row['cmse']}<br />LPIPS: {row['lpips']}<br />LPIPS_KFS: {row['lpipskfs']}<br />LPIPS_MIX: {row['lpipsmix']}<br />LPIPKFS/LPIPS: {row[col]}</td>")
            print("</tr>")
        print("</table>")

print("<h1>Largest lpips_kfs / lpips</h1>")
printout(largest)

print("<h1>Smallest lpips_kfs / lpips</h1>")
printout(smallest)
