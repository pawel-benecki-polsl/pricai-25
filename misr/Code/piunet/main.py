import os
import shutil
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import Mus2DatasetTrain, Mus2DatasetVal
from losses import cmse, custom_metric_model_shifts_from_cpsnr, \
    l1_registered_loss, l1_registered_uncertainty_loss
from metrics.models import load_lpips, Models
from model import PIUNET

trained_models = {
    "mus2_suma_l1_lpips": "30/0096a75d2b0241f29cb9724634415bd3",
    "mus2_suma_l1_lpips_kfs_mix": "29/e32974e09a834e6e9b7d397ebe189d90",
    "mus2_suma_l1_lpips_kfs": "27/b5b1ff49585b48f0931e6fd8b27884d2",

    "mus2_lpips_1vote_kfs_2_votes_human_dotrenowane_l1": "26/cc9e6b3606674d4fac0037599e8511fe",
    "mus2_l1_dotrenowane_lpips_1vote_kfs_2_votes_human": "25/8ec9cfd1b77242739d74c6f260be0449",

    "MuS2_L1": "22/5a4aa172253547afa3f82356992db7cd",
    "MuS2_CPSNR": "23/a273af954fd248ecabcf10aaf56e2ea6",
    "MuS2_LPIPS": "20/a6499ccea9954b53a16401591123b7a3",
    "MuS2_LPIPS_KFS": "21/914420557021413696377f1c6d774110",
    "mus2_lpips_1vote_kfs_2_votes_human": "24/90ce91c49800450f94e3463336fc201c",
}


class LossBase:

    def loss(self, epoch, ref, sr, sigma_sr, mask, size):
        raise "not implemented"


class LossL1(LossBase):
    def __str__(self):
        return "L1"

    def loss(self, epoch, ref, sr, sigma_sr, mask, size):
        if epoch < 1:
            loss = l1_registered_loss(ref, sr, mask, size)
        else:
            loss = l1_registered_uncertainty_loss(ref, sr, sigma_sr, mask, size)
        return loss


class LossL2(LossBase):
    def __str__(self):
        return "L2"

    def loss(self, epoch, ref, sr, sigma_sr, mask, size):
        return cmse(ref, sr, mask, size)


class LossLpipsBase(LossBase):
    def __init__(self, model):
        self.model = model

    def loss(self, epoch, ref, sr, sigma_sr, mask, size):
        return custom_metric_model_shifts_from_cpsnr(ref, sr, mask, size, self.model)


class LossLpips(LossLpipsBase):
    def __str__(self):
        return "LPIPS"

    def __init__(self):
        super().__init__(load_lpips())


class LossLpipsKFS(LossLpipsBase):
    def __str__(self):
        return "LPIPS_KFS"

    def __init__(self):
        super().__init__(load_lpips(Models.LPIPS_KFS_20221206))


class LossLpipsMix(LossLpipsBase):
    def __str__(self):
        return "LPIPS_MIX"

    def __init__(self):
        super().__init__(load_lpips(Models.LPIPS_1VOTE_KFS_2VOTES_HUMAN))


class LossSum(LossBase):
    def __init__(self, first: LossBase, second: LossBase):
        self.first = first
        self.second = second

    def __str__(self):
        return f"{self.first}_plus_{self.second}"

    def loss(self, epoch, ref, sr, sigma_sr, mask, size):
        return self.first.loss(epoch, ref, sr, sigma_sr, mask, size) \
               + self.second.loss(epoch, ref, sr, sigma_sr, mask, size)


class ValLossCollector:
    def __init__(self, loss: LossBase):
        self.loss = loss
        self.min = 1E9
        self.collected = []

    def start_epoch(self):
        self.collected = []

    def tick(self, value):
        self.collected.append(value.cpu().detach().numpy())

    def collect(self):
        mean = np.mean(self.collected)
        if mean < self.min:
            self.min = mean
            return True, mean
        return False, mean


if __name__ == '__main__':
    lpips = LossLpips()
    lpips_kfs = LossLpipsKFS()
    lpips_mix = LossLpipsMix()
    l1 = LossL1()
    l2 = LossL2()

    the_loss = lpips
    model_desc = "mus2_l1_dotrenowane_lpips"
    conf_experiment = f"PIUNET {model_desc}"

    exp, run = trained_models['MuS2_L1'].split('/')
    start_model_path = f"c:\\Users\\pbenecki\\mlflow\\mlruns\\{exp}\\{run}\\artifacts\\model_weights_newest.pt"

    conf_output_dir = f"c:\\experiments\\202305\\{model_desc}"
    os.makedirs(conf_output_dir, exist_ok=True)
    ##################################
    ##################################
    ##################################

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment(conf_experiment)
    mlflow.start_run(run_name=model_desc)
    run = mlflow.active_run()
    artifacts_location = os.path.join("c:\\users\\pbenecki\\mlflow", run.info.artifact_uri)

    shutil.copy(__file__, os.path.join(artifacts_location, "main.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "config.py"), os.path.join(artifacts_location, "config.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "losses.py"), os.path.join(artifacts_location, "losses.py"))

    mlflow.log_param("output_dir", conf_output_dir)

    model_time = time.strftime("%Y%m%d_%H%M")
    ##################################
    ##################################
    ##################################

    # Import config
    config = Config()

    # Import datasets
    # train_dataset = ProbaVDatasetTrain(config)
    train_dataset = Mus2DatasetTrain(config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               drop_last=False, num_workers=config.workers)

    # val_dataset = ProbaVDatasetVal(config)
    val_dataset = Mus2DatasetVal(config)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    # dataset_mu = torch.Tensor((train_dataset.mu,)).to(config.device)
    # dataset_sigma = torch.Tensor((train_dataset.sigma,)).to(config.device)

    # Create model
    model = PIUNET(config, None)
    if start_model_path is not None:
        problems = model.load_state_dict(torch.load(start_model_path))
        print(problems)
    model.cuda()

    # print('No. params: %d' % (sum(p.numel() for p in model.parameters() if p.requires_grad),))
    #
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150000], gamma=0.2, last_epoch=-1)

    tot_steps = 0

    val_collectors = [
        ValLossCollector(l1),
        ValLossCollector(l2),
        ValLossCollector(the_loss),
        ValLossCollector(lpips_mix),
        ValLossCollector(lpips),
        ValLossCollector(lpips_kfs),
    ]


    def _to_8bit(to_save):
        to_save = to_save.cpu().detach().numpy()[0, 0, :, :]
        return ((to_save - to_save.min()) * (1 / (to_save.max() - to_save.min()) * 255)).astype('uint8')


    for epoch in range(config.N_epoch):
        for step, (tr_lr, tr_hr, tr_mask) in enumerate(tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()

            tr_lr = torch.Tensor(tr_lr).to(config.device)
            tr_hr = torch.Tensor(tr_hr).to(config.device)
            tr_mask = torch.Tensor(tr_mask).to(config.device)

            mu_sr, sigma_sr = model(tr_lr)

            loss = the_loss.loss(epoch, tr_hr, mu_sr, sigma_sr, tr_mask, config.patch_size * 3)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 15)

            optimizer.step()
            scheduler.step()

            if step % config.log_every_iter == 0:
                loss_value = loss.cpu().detach().numpy()

                # Image.fromarray(_to_8bit(mu_sr)).save(
                #     f"{conf_output_dir}\\train_epoch({str(epoch).zfill(3)})_img({str(step).zfill(6)})_cpsnr({cpsnr_value})_loss({loss_value}).png")

                mlflow.log_metric("train.loss", loss_value, tot_steps + step)

        tot_steps = tot_steps + step

        if config.validate:
            model.eval()
            with torch.no_grad():
                for lc in val_collectors:
                    lc.start_epoch()

                x_sr_all = []
                x_hr_all = []
                s_sr_all = []

                for val_step, (val_lr, val_hr, val_mask) in enumerate(tqdm(val_loader)):
                    val_lr = torch.Tensor(val_lr).to(config.device)
                    # Image.fromarray(_to_8bit(val_lr)).save(
                    #     f"{conf_output_dir}\\lr0({str(val_step).zfill(3)}).png")
                    val_hr = torch.Tensor(val_hr).to(config.device)
                    val_mask = torch.Tensor(val_mask).to(config.device)

                    mu_sr, sigma_sr = model(val_lr)

                    # mu_sr = mu_sr * dataset_sigma + dataset_mu
                    # sigma_sr = sigma_sr + torch.log(dataset_sigma)

                    x_sr_all.append(mu_sr)
                    x_hr_all.append(val_hr)
                    s_sr_all.append(sigma_sr)

                    for lc in val_collectors:
                        lc_here = lc.loss.loss(epoch, val_hr, mu_sr, sigma_sr, val_mask, config.patch_size * 3)
                        lc.tick(lc_here)

                    # Image.fromarray(_to_8bit(mu_sr)).save(
                    #     f"{conf_output_dir}\\epoch({str(epoch).zfill(3)})_img({str(val_step).zfill(3)})_psnr({psnr_here})_lpips({lpips_here})_lpipskfs({lpips_kfs_here}).png")
                    # Image.fromarray(_to_8bit(val_hr)).save(
                    #     f"{conf_output_dir}\\hr({str(val_step).zfill(3)}).png")

                for lc in val_collectors:
                    new_best, val = lc.collect()
                    mlflow.log_metric(f"val.{lc.loss}", val, tot_steps)
                    if new_best:
                        torch.save(model.state_dict(),
                                   os.path.join(artifacts_location, f"model_weights_best_{lc.loss}.pt"))

        torch.save(model.state_dict(),
                   os.path.join(artifacts_location, f"model_weights_epoch_{str(epoch).zfill(3)}.pt"))

    mlflow.end_run()
