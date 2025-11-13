import glob
import os
import pandas as pd
from mus2.metrics_single_band import MetricsSingleBand

base_dir = "a:\\phd_artifacts"

for dataset in [
    # "__probaRED_results", "__probaNIR_results", "__harvard_results",
    "__probaRED_results",
    "__probaNIR_results",
    "__20230607_probaNIR_interp",
    "__20230607_probaRED_interp"
    # "__mus2_200px_results"
]:
    resultss = []
    for file in glob.glob(os.path.join(base_dir, dataset, "details_other_8bit.csv")):
        df = pd.read_csv(file)

        ddd = os.path.splitext(os.path.basename(file))[0]

        columns = ['dataset', 'model', "PSNR_matched", "PSNR", "SSIM_matched", "SSIM", "LPIPS_matched",
                   "LPIPS", "LPIPS_KFS_matched", "LPIPS_KFS", "LPIPS_MIX_matched", "LPIPS_MIX", "cPSNR_matched",
                   "cPSNR", "cSSIM_matched", "cSSIM"]
        df = df[columns]

        # Wykonaj agregację
        aggregated_data = df.groupby(['dataset', 'model']).agg(['mean', 'min', 'max', 'std'])

        aggregated_data.to_csv(os.path.join(base_dir, dataset, f"summary_{ddd}.csv"))
        print(file)
