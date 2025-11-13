import glob
import os

import numpy as np
import pandas as pd
from skimage.exposure import match_histograms

from mus2.metrics_single_band import MetricsSingleBand

base_dir = "a:\\phd_artifacts"

metrics = MetricsSingleBand()

for dataset in [
    "__harvard_results",
    # "__probaRED_results",
    # "__probaNIR_results",
    # "__20230607_probaNIR_interp",
    # "__20230607_probaRED_interp"
]:
    resultss = []
    for model in os.listdir(os.path.join(base_dir, dataset)):
        print(f"{dataset}:{model}")
        hrs = sorted(glob.glob(os.path.join(base_dir, dataset, model, "hr_img(*).npy")))
        srs = sorted(glob.glob(os.path.join(base_dir, dataset, model, "sr_img(*).npy")))


        def do_index(files):
            file_index = {}
            for file in files:
                index = os.path.splitext(os.path.basename(file))[0].split("(")[1].split(")")[0]
                filename = os.path.basename(file)
                file_index[index] = (filename, file)
            return file_index


        hrs_index = do_index(hrs)
        srs_index = do_index(srs)


        def _to_8bit(input):
            arr = input[6:-6,6:-6]
            return ((input - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')


        for key in hrs_index:
            hr_filename, hr_filepath = hrs_index[key]
            sr_filename, sr_filepath = srs_index[key]

            hr_orig = np.load(hr_filepath)[0, 0, :, :].astype('float32')
            sr_orig = np.load(sr_filepath).astype('float32')

            # from dataset import Mus2Config as CFG
            from dataset import HarvardConfig as CFG
            hr_8bit = (hr_orig * CFG.y_std + CFG.y_mean)
            sr_8bit = (sr_orig * CFG.y_std + CFG.y_mean)
            # proba_mu = 7433.6436
            # proba_sigma = 2353.0723
            #
            # hr_8bit = (hr_orig * proba_sigma + proba_mu) / 16384.0 * 255.0
            # sr_8bit = (sr_orig * proba_sigma + proba_mu) / 16384.0 * 255.0


            diff = hr_8bit.astype('float32') - sr_8bit.astype('float32')

            sr_matched = match_histograms(sr_8bit, hr_8bit).astype('uint8')

            diff2 = hr_8bit.astype('float32') - sr_matched.astype('float32')

            results = {"dataset": dataset, "model": model, "img": int(key)}
            for metric, func in metrics.metrics.items():
                results[f"{metric}_matched"] = func(sr_matched, hr_8bit, None)
                results[metric] = func(sr_8bit, hr_8bit, None)
            resultss.append(results)

    df = pd.DataFrame(resultss)
    df.to_csv(os.path.join(base_dir, dataset, "details_other_8bit.csv"))
