import os

import numpy as np
import torch
import torch.nn.functional as func
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from config import Config, ConfigNIR, ConfigRED
from dataset import ProbaVDatasetVal, Mus2DatasetTest, \
    HarvardDatasetTest, Mus2Dataset200pxLR
from losses import custom_metric_model_shifts_from_cpsnr
from metrics.models import load_lpips, Models
from model import PIUNET

trained_models = {
    # "mus2_suma_l1_lpips": "30/0096a75d2b0241f29cb9724634415bd3",
    "mus2_suma_L1_LPIPS": "30/77733f43641b4f91b79a44540ccf5c14",
    "mus2_suma_L1_LPIPS_MIX": "29/e32974e09a834e6e9b7d397ebe189d90",
    "mus2_suma_L1_LPIPS_KFS": "27/b5b1ff49585b48f0931e6fd8b27884d2",

    "mus2_sekw_LPIPS_MIX_dotrenowane_L1": "26/cc9e6b3606674d4fac0037599e8511fe",
    "mus2_sekw_L1_dotrenowane_LPIPS_MIX": "25/8ec9cfd1b77242739d74c6f260be0449",

    "mus2_sekw_LPIPS_dotrenowane_L1": "32/c7574b7cee1b4475a2e37041e79e6780",
    "mus2_sekw_L1_dotrenowane_LPIPS": "35/3e9dbebab8044583ba5e9668c7a076ba",

    "mus2_sekw_LPIPS_KFS_dotrenowane_L1": "33/734fc85f655a458aa6655ab48b1ae749",
    "mus2_sekw_L1_dotrenowane_LPIPS_KFS": "34/1ccb487506d24dae89f7478282d6a9f3",

    "muS2_L1": "22/5a4aa172253547afa3f82356992db7cd",
    "muS2_CPSNR": "23/a273af954fd248ecabcf10aaf56e2ea6",
    "muS2_LPIPS": "20/a6499ccea9954b53a16401591123b7a3",
    "muS2_LPIPS_KFS": "21/914420557021413696377f1c6d774110",
    "mus2_LPIPC_MIX": "24/90ce91c49800450f94e3463336fc201c",
}

model_metric_lpips = load_lpips()
model_metric_lpips_mix = load_lpips(Models.LPIPS_1VOTE_KFS_2VOTES_HUMAN)
model_metric_lpips_kfs = load_lpips(Models.LPIPS_KFS_20221206)

lpips_model_ref = custom_metric_model_shifts_from_cpsnr


def _to_8bit(to_save):
    to_save = to_save.cpu().detach().numpy()[0, 0, :, :]
    return ((to_save - to_save.min()) * (1 / (to_save.max() - to_save.min()) * 255)).astype('uint8')


def execute_it(config, the_folder, exp_name, model_path, interpolation=None):
    print(model_path)

    if interpolation is None:
        model = PIUNET(config, None)
        problems = model.load_state_dict(torch.load(model_path))
        print(problems)
        model.cuda()

    output_dir = f"a:\\phd_artifacts\\{the_folder}\\{exp_name}"
    os.makedirs(output_dir, exist_ok=True)


    import dataset as ds

    if interpolation is None:
        model.eval()
    with torch.no_grad():
        for test_step, (test_lr, test_hr, test_mask) in enumerate(tqdm(test_loader)):
            save_path_lr_img = f"{output_dir}\\lr0_img({str(test_step).zfill(4)}).png"
            save_path_lr_np = f"{output_dir}\\lr_img({str(test_step).zfill(4)}).npy"

            save_path_hr_img = f"{output_dir}\\hr_img({str(test_step).zfill(4)}).png"
            save_path_hr_np = f"{output_dir}\\hr_img({str(test_step).zfill(4)}).npy"

            save_path_sr_img = f"{output_dir}\\sr_img({str(test_step).zfill(4)}).png"
            save_path_sr_np = f"{output_dir}\\sr_img({str(test_step).zfill(4)}).npy"

            test_hr = torch.Tensor(test_hr).to(config.device)
            Image.fromarray(_to_8bit(test_hr)).save(save_path_hr_img)
            hr_file_size = os.path.getsize(save_path_hr_img)
            if hr_file_size < 5000:  # eliminates most of the sea-only patches, which artificially increase PSNR
                os.remove(save_path_hr_img)
                continue
            np.save(save_path_hr_np, test_hr.cpu().detach().numpy())

            hr_size = test_hr.shape[2], test_hr.shape[3]

            test_lr = torch.Tensor(test_lr).to(config.device)
            if interpolation is None:
                mu_sr, sigma_sr = model(test_lr)
            elif interpolation == "NN":
                the_mean = torch.mean(test_lr, dim=1, keepdim=True)

                mu_sr = func.interpolate(the_mean, size=hr_size , mode='nearest')
            elif interpolation == "bilinear":
                the_mean = torch.mean(test_lr, dim=1, keepdim=True)
                mu_sr = func.interpolate(the_mean, size=hr_size, mode='bilinear', align_corners=True)
            elif interpolation == "bicubic":
                the_mean = torch.mean(test_lr, dim=1, keepdim=True)
                mu_sr = func.interpolate(the_mean, size=hr_size, mode='bicubic', align_corners=True)


            Image.fromarray(_to_8bit(test_lr)).save(save_path_lr_img)
            np.save(save_path_lr_np, test_lr.cpu().detach().numpy()[0, 0, :, :])

            Image.fromarray(_to_8bit(mu_sr)).save(save_path_sr_img)
            np.save(save_path_sr_np, mu_sr.cpu().detach().numpy()[0, 0, :, :])


def get_model_path(art_path):
    if os.path.exists(os.path.join(art_path, "model_weights_newest.pt")):
        return os.path.join(art_path, "model_weights_newest.pt")
    return os.path.join(art_path, sorted([f for f in os.listdir(art_path) if f.startswith("model_weights_epoch_")])[-1])


if __name__ == '__main__':
    for the_folder, config, val_dataset in [
        # ("__20230607_mus2", Config(), Mus2DatasetTest(Config())),
        ("__20230607_harvard_interp", Config(), HarvardDatasetTest(Config())),
        ("__20230607_probaNIR_interp", Config(), ProbaVDatasetVal(ConfigNIR())),
        ("__20230607_probaRED_interp", Config(), ProbaVDatasetVal(ConfigRED())),
    ]:
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        execute_it(config, the_folder, "NN", None, "NN")
        execute_it(config, the_folder, "bilinear", None, "bilinear")
        execute_it(config, the_folder, "bicubic", None, "bicubic")
        # execute_it(config, the_folder, "orig_l1_nir", "a:\\phd_artifacts\\nir_model_checkpoint.pt")
        # execute_it(config, the_folder, "orig_l1_red", "a:\\phd_artifacts\\red_model_checkpoint.pt")

        # for exp_name, exp_id_run_id in trained_models.items():
        #     exp_id, run_id = exp_id_run_id.split('/')
        #     model_path = get_model_path(f"a:\\phd_artifacts\\{exp_id}\\{run_id}\\artifacts")
        #
        #     execute_it(config, the_folder, exp_name, model_path)
