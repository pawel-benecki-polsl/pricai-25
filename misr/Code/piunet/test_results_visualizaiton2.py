import os
import re

import numpy as np
import pandas as pd
import pandas as pd

the_folder = "__mus2_results"
# Example usage: find files in current directory and its subdirectories

details = pd.read_csv(f"a:\\phd_artifacts\\{the_folder}\\details_other_8bit.csv")
# col = 'lpips_kfsDIVlpips'
# details[col] = details['lpipskfs'] / details['lpips']

# details = details.sort_values(col)
# largest = details.tail(50)
# smallest = details.head(50)



# dataset,model,img,PSNR_matched,PSNR,SSIM_matched,SSIM,LPIPS_matched,LPIPS,LPIPS_KFS_matched,LPIPS_KFS,LPIPS_MIX_matched,LPIPS_MIX,cPSNR_matched,cPSNR,cSSIM_matched,cSSIM

def printout(filtered):
    # for img_number, vall in zip(img_numbers, vals):
    #     filtered = details[details['img'] == img_number] and details[details[col] == vall]
        print("<html><head>"
              "<style>  "
              " tr.bottomborder {    border-bottom: 2px solid black;  } "
              " .vt {      writing-mode: vertical-rl;      transform: rotate(180deg);    }"
              "</style><"
              "/head><body>")
        print("<table>")
        models = filtered['model'].unique()
        print("<tr><td class='vt'>LR</td><td class='vt'>HR</td>", end='')
        for model in models:
            print(f"<td class='vt'>{model}</td>", end='')
        print("</tr>")
        for idx, row in filtered[filtered['model'] == 'muS2_CPSNR'].iterrows():
            if idx % 30 != 0:
                continue
            print("<tr>")
            lr_path = f"a:\\phd_artifacts\\{the_folder}\\muS2_CPSNR\\lr0_img({str(row['img']).zfill(4)}).png"
            hr_path = f"a:\\phd_artifacts\\{the_folder}\\muS2_CPSNR\\hr_img({str(row['img']).zfill(4)}).png"
            print(f"<td><img src='{lr_path}'></td>")
            print(f"<td><img src='{hr_path}'></td>")

            all_images_with_the_img = filtered[filtered['img'] == row['img']]

            imgs = dict()
            for model in all_images_with_the_img['model'].unique():
                # Do something with each model
                imgs[model] = all_images_with_the_img[all_images_with_the_img['model'] == model]

            def img_src(df):
                return f"a:\\phd_artifacts\\{the_folder}\\{df['model'].iloc[0]}\\sr_img({str(row['img']).zfill(4)}).png"

            for k, v in imgs.items():
                print(f"<td><img src='{img_src(v)}'/></td>")

            print("</tr>")

            # def vals(df):
            #     return f"cMSE:{df['cmse'].iloc[0]:.6f}<br />lpips:{df['lpips'].iloc[0]:.6f}<br />kfs:{df['lpipskfs'].iloc[0]:.6f}<br />mix:{df['lpipsmix'].iloc[0]:.6f}<br /><br /><br />"
            #
            # print(f"<tr class=\"bottomborder\">"
            #       f"<td>{vals(l2_img)}</td>"
            #       f"<td>{vals(l1_img)}</td>"
            #       f"<td>IDX:{idx}</td>"
            #       f"<td>{vals(lpips_then_l1_img)}</td>"
            #       f"<td>{vals(lpips_kfs_then_l1_img)}</td>"
            #       f"<td>{vals(lpips_sum_l1)}</td>"
            #       f"<td>{vals(lpips_kfs_sum_l1)}</td>"
            #       f"<td>{vals(probaNIR)}</td>"
            #       f"</tr>"
            #       )
        print("</table></body></html>")


def printout2(filtered):
    # for img_number, vall in zip(img_numbers, vals):
    #     filtered = details[details['img'] == img_number] and details[details[col] == vall]
        print("<html><head>"
              "<style>  "
              " tr.bottomborder {    border-bottom: 2px solid black;  } "
              " .vt {      writing-mode: vertical-rl;      transform: rotate(180deg);    }"
              "</style><"
              "/head><body>")
        print("<table>")
        # models = filtered['model'].unique()
        # print("<tr><td class='vt'>LR</td><td class='vt'>HR</td>", end='')
        # for model in models:
        #     print(f"<td class='vt'>{model}</td>", end='')
        # print("</tr>")
        for idx, row in filtered[filtered['model'] == 'muS2_CPSNR'].iterrows():
            if row['img'] not in [0, 120, 243, 423]:
                continue
            # if idx % 30 != 0:
            #     continue
            lr_path = f"a:\\phd_artifacts\\{the_folder}\\muS2_CPSNR\\lr0_img({str(row['img']).zfill(4)}).png"
            hr_path = f"a:\\phd_artifacts\\{the_folder}\\muS2_CPSNR\\hr_img({str(row['img']).zfill(4)}).png"

            all_images_with_the_img = filtered[filtered['img'] == row['img']]

            imgs = dict()
            for model in all_images_with_the_img['model'].unique():
                # Do something with each model
                imgs[model] = all_images_with_the_img[all_images_with_the_img['model'] == model]

            def img_src(df):
                return f"a:\\phd_artifacts\\{the_folder}\\{df['model'].iloc[0]}\\sr_img({str(row['img']).zfill(4)}).png"

            def load_shrink_save(src, how_much, dst):
                from PIL import Image
                img = Image.open(src)
                width, height = img.size
                # img = img.resize((width*3, height*3), Image.NEAREST)

                img = img.crop(((how_much), (how_much), (width - how_much), (height - how_much)))
                where_save = os.path.join("c:\\projekty\\phd\\img_experiments\\results_mus2", dst)
                img.save(where_save)
                print(f"Image {src} has been shrunk and saved to {where_save}")

            # print("<tr>")
            # print(f"<td><img src='{img_src(imgs['muS2_LPIPS'])}'/><br/>LPIPS</td>")
            # load_shrink_save(img_src(imgs['muS2_LPIPS']), 5, f"{row['img']:04d}_1_1.png")
            # print(f"<td><img src='{img_src(imgs['muS2_LPIPS_KFS'])}'/><br/>LPIPS_KFS</td>")
            # load_shrink_save(img_src(imgs['muS2_LPIPS_KFS']), 5, f"{row['img']:04d}_1_2.png")
            # print(f"<td><img src='{img_src(imgs['mus2_LPIPC_MIX'])}'/><br/>LPIPS_MIX</td>")
            # load_shrink_save(img_src(imgs['mus2_LPIPC_MIX']), 5, f"{row['img']:04d}_1_3.png")
            # print("</tr>")
            #
            # print("<tr>")
            # print(f"<td><img src='{img_src(imgs['mus2_sekw_LPIPS_dotrenowane_L1'])}'/><br/>LPIPS, L1</td>")
            # load_shrink_save(img_src(imgs['mus2_sekw_LPIPS_dotrenowane_L1']), 5, f"{row['img']:04d}_2_1.png")
            # print(f"<td><img src='{img_src(imgs['mus2_sekw_LPIPS_KFS_dotrenowane_L1'])}'/><br/>LPIPS_KFS, L1</td>")
            load_shrink_save(lr_path, 5, f"{row['img']:04d}_2_2_lr.png")
            # # load_shrink_save(img_src(imgs['mus2_sekw_LPIPS_KFS_dotrenowane_L1']), 5, f"{row['img']:04d}_2_2.png")
            # print(f"<td><img src='{img_src(imgs['mus2_sekw_LPIPS_MIX_dotrenowane_L1'])}'/><br/>LPIPS_MIX, L1</td>")
            # load_shrink_save(img_src(imgs['mus2_sekw_LPIPS_MIX_dotrenowane_L1']), 5, f"{row['img']:04d}_2_3.png")
            # print("</tr>")
            #
            # print("<tr>")
            # print(f"<td><img src='{img_src(imgs['muS2_CPSNR'])}'/><br/>cPSNR</td>")
            # load_shrink_save(img_src(imgs['muS2_CPSNR']), 5, f"{row['img']:04d}_3_1.png")
            # print(f"<td><img src='{hr_path}'/><br /><b>HR</b></td>")
            # load_shrink_save(hr_path, 5, f"{row['img']:04d}_3_2.png")
            # print(f"<td><img src='{img_src(imgs['muS2_L1'])}'/><br/>L1</td>")
            # load_shrink_save(img_src(imgs['muS2_L1']), 5, f"{row['img']:04d}_3_3.png")
            # print("</tr>")
            #
            # print("<tr>")
            # print(f"<td><img src='{img_src(imgs['mus2_suma_L1_LPIPS'])}'/><br/>L1+LPIPS</td>")
            # load_shrink_save(img_src(imgs['mus2_suma_L1_LPIPS']), 5, f"{row['img']:04d}_4_1.png")
            # print(f"<td><img src='{img_src(imgs['mus2_suma_L1_LPIPS_KFS'])}'/><br/>L1+DKFS</td>")
            # load_shrink_save(img_src(imgs['mus2_suma_L1_LPIPS_KFS']), 5, f"{row['img']:04d}_4_2.png")
            # print(f"<td><img src='{img_src(imgs['mus2_suma_L1_LPIPS_MIX'])}'/><br/>L1+DKFS_MIX</td>")
            # load_shrink_save(img_src(imgs['mus2_suma_L1_LPIPS_MIX']), 5, f"{row['img']:04d}_4_3.png")
            # print("</tr>")

        print("</table></body></html>")
printout2(details)

