import numpy as np
import pytorch_ssim
import skimage.registration
import torch
import torchvision.transforms as T
from torchvision.transforms import Normalize, GaussianBlur


def load_trained_metric_model(path: str):
    from model import PatternModel
    metric_model_path = "C:\\projekty\\superdeep\\scripts\\lpips_trainer\\checkpoints-KFS_SIFT\\best-checkpoint.ckpt"
    mm = PatternModel.load_from_checkpoint(metric_model_path)
    mm.eval()
    for param in mm.model.parameters():
        param.requires_grad = False
    return mm


def custom_metric_model(y_true, y_pred, y_mask, HR_SIZE, metric_model, step):
    '''
    y_true, y_pred, y_mask - [batch, ignored, width, height]
    '''
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    tfm = Normalize([.5, .5, .5], [.5, .5, .5])

    y_true = torch.stack([y_true, y_true, y_true], dim=1)
    y_pred = torch.stack([y_pred, y_pred, y_pred], dim=1)
    y_mask = torch.stack([y_mask, y_mask, y_mask], dim=1)

    border = 2
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    cropped_predictions = tfm(y_pred[:, :, border:size_image - border, border:size_image - border])

    X = []
    for i in range(max_pixels_shifts + 1):
        for j in range(max_pixels_shifts + 1):
            h_crop = i + (size_image - max_pixels_shifts)
            w_crop = j + (size_image - max_pixels_shifts)

            cropped_labels = tfm(y_true[:, :, i:h_crop, j:w_crop])
            cropped_y_mask = y_mask[:, :, i:h_crop, j:w_crop]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            batch_losses = []
            for k in range(cropped_labels_masked.shape[0]):
                batch_element_loss_value = metric_model(cropped_labels_masked[k], cropped_predictions_masked[k])
                batch_losses.append(torch.flatten(batch_element_loss_value))

            X.append(torch.stack(batch_losses))
    X = torch.stack(X)
    min_l1, indxs = torch.min(X, dim=0)
    print(f"{step},{indxs.detach().cpu().numpy().flatten()}")

    return torch.mean(min_l1)


def high_pass_filter_mse(y_true, y_pred, y_mask, HR_SIZE):
    '''
    y_true, y_pred, y_mask - [batch, ignored, width, height]
    '''
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    tfm = T.Normalize([.5], [.5])
    blur = T.GaussianBlur(kernel_size=9, sigma=1.5)

    border = 2
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    cropped_predictions = tfm(y_pred[:, border:size_image - border, border:size_image - border])

    X = []
    for i in range(max_pixels_shifts + 1):
        for j in range(max_pixels_shifts + 1):
            h_crop = i + (size_image - max_pixels_shifts)
            w_crop = j + (size_image - max_pixels_shifts)

            cropped_labels = tfm(y_true[:, i:h_crop, j:w_crop])
            cropped_y_mask = y_mask[:, i:h_crop, j:w_crop]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            blurred_pred = blur(cropped_predictions_masked)
            blurred_labels = blur(cropped_labels_masked)

            hf_pred = (torch.abs(cropped_predictions_masked - blurred_pred))
            hf_label = (torch.abs(cropped_labels_masked - blurred_labels))

            l2_loss = torch.sum(torch.square(hf_pred - hf_label), dim=(1, 2)) / total_pixels_masked

            X.append(l2_loss)
    X = torch.stack(X)
    min_l1, indxs = torch.min(X, dim=0)

    return torch.mean(min_l1)


def shifts_as_cpsnr(y_true, y_pred, y_mask, HR_SIZE):
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(1, 2))
            # print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            l1_loss = torch.sum(torch.abs(cropped_labels_masked - corrected_cropped_predictions),
                                dim=(1, 2)) / total_pixels_masked

            X.append(l1_loss)
    X = torch.stack(X)
    _, indxs = torch.min(X, dim=0)
    indxs = indxs.detach().cpu().numpy()
    shift = (max_pixels_shifts / 2)
    dims = (max_pixels_shifts + 1)
    return [(idx % dims - shift, idx // dims - shift) for idx in indxs]


def custom_metric_model_shifts_from_cpsnr(y_true, y_pred, y_mask, HR_SIZE, metric_model):
    shifts = shifts_as_cpsnr(y_true, y_pred, y_mask, HR_SIZE)
    '''
    y_true, y_pred, y_mask - [batch, ignored, width, height]
    '''
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    batch_losses = []

    tfm = Normalize([0, 0, 0], [.5, .5, .5])
    # tfm = Normalize([.5, .5, .5], [.5, .5, .5])

    border = 2

    for no, (yt, yp, m) in enumerate(zip(y_true, y_pred, y_mask)):
        m = torch.stack([m, m, m])[:, border:-border]
        yt = torch.stack([yt, yt, yt])[:, border:-border] * m

        yp = torch.roll(yp, (int(shifts[no][0]), int(shifts[no][1])), dims=(0, 1))
        yp = torch.stack([yp, yp, yp])[:, border:-border] * m

        yt = tfm(yt)
        yp = tfm(yp)

        batch_element_loss_value = metric_model(yt, yp)
        batch_losses.append(torch.flatten(batch_element_loss_value))

    losses = torch.stack(batch_losses)
    return torch.mean(losses)


def custom_metric_model_shifts_from_skimage(y_true, y_pred, y_mask, HR_SIZE, metric_model, step):
    '''
    y_true, y_pred, y_mask - [batch, ignored, width, height]
    '''
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    batch_losses = []

    tfm = Normalize([.5, .5, .5], [.5, .5, .5])

    border = 2

    for yt, yp, m in zip(y_true, y_pred, y_mask):
        yt_np = yt.detach().cpu().numpy()
        yp_np = yp.detach().cpu().numpy()
        m_np = m.detach().cpu().numpy()
        s = skimage.registration.phase_cross_correlation(yt_np, yp_np, reference_mask=m_np)

        m = torch.stack([m, m, m])[:, border:-border]
        yt = torch.stack([yt, yt, yt])[:, border:-border] * m

        yp = torch.roll(yp, (int(round(s[0])), int(round(s[1]))), dims=(0, 1))
        yp = torch.stack([yp, yp, yp])[:, border:-border] * m

        yt = tfm(yt)
        yp = tfm(yp)

        batch_element_loss_value = metric_model(yt, yp)
        batch_losses.append(torch.flatten(batch_element_loss_value))

    losses = torch.stack(batch_losses)
    return torch.mean(losses)


def l1_registered_loss(y_true, y_pred, y_mask, HR_SIZE):
    y_true = y_true[:, 0, :, :]
    y_pred = y_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(1, 2))
            # print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            l1_loss = torch.sum(torch.abs(cropped_labels_masked - corrected_cropped_predictions),
                                dim=(1, 2)) / total_pixels_masked

            X.append(l1_loss)
    X = torch.stack(X)
    min_l1, indxs = torch.min(X, dim=0)

    return torch.mean(min_l1)


def l1_registered_uncertainty_loss(y_true, mu_pred, sigma_pred, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """

    y_true = y_true[:, 0, :, :]
    mu_pred = mu_pred[:, 0, :, :]
    sigma_pred = sigma_pred[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions_mu = mu_pred[:, border:size_image - border, border:size_image - border]
    cropped_predictions_sigma = sigma_pred[:, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_mu_masked = cropped_predictions_mu * cropped_y_mask
            cropped_predictions_sigma_masked = cropped_predictions_sigma * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_mu_masked,
                                                        dim=(1, 2))
            # print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions_mu = cropped_predictions_mu_masked + b
            corrected_cropped_predictions_mu = corrected_cropped_predictions_mu * cropped_y_mask
            corrected_cropped_predictions_sigma = cropped_predictions_sigma_masked + b
            corrected_cropped_predictions_sigma = corrected_cropped_predictions_sigma * cropped_y_mask

            # l1_loss = torch.sum(torch.abs(cropped_labels_masked-corrected_cropped_predictions), dim=(1,2))/total_pixels_masked

            y = cropped_labels_masked
            m = corrected_cropped_predictions_mu
            s = corrected_cropped_predictions_sigma
            # l1_loss = torch.sum( torch.log(2*s) + torch.abs(y-m)/s, dim=(1,2))/total_pixels_masked
            l1_loss = torch.sum(s + torch.abs(y - m) * torch.exp(-s), dim=(1, 2)) / total_pixels_masked

            X.append(l1_loss)
    X = torch.stack(X)
    min_l1, indxs = torch.min(X, dim=0)
    # print(f"{step},{indxs.detach().cpu().numpy().flatten()}")

    return torch.mean(min_l1)


def NIG_NLL(y, gamma, v, alpha, beta, mask, reduced=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi) - torch.log(v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    nll = nll * mask

    return torch.mean(nll) if reduced else nll


def evidential_loss(y_true, gamma, v, alpha, beta, reduced=True, coeff=1.0):
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduced)
    # loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    # return loss_nll + coeff * loss_reg
    return loss_nll


def registered_evidential_loss(y_true, gamma, v, alpha, beta, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """

    y_true = y_true[:, 0, :, :]
    gamma = gamma[:, 0, :, :]
    v = v[:, 0, :, :]
    alpha = alpha[:, 0, :, :]
    beta = beta[:, 0, :, :]
    y_mask = y_mask[:, 0, :, :]

    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image

    cropped_predictions_gamma = gamma[:, border:size_image - border, border:size_image - border]
    cropped_predictions_v = v[:, border:size_image - border, border:size_image - border]
    cropped_predictions_alpha = alpha[:, border:size_image - border, border:size_image - border]
    cropped_predictions_beta = beta[:, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(1, 2))

            cropped_predictions_gamma_masked = cropped_predictions_gamma * cropped_y_mask
            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_gamma_masked,
                                                        dim=(1, 2))
            # print(b.shape)
            b = torch.reshape(b, (y_shape[0], 1, 1))

            corrected_cropped_predictions_gamma = cropped_predictions_gamma + b
            corrected_cropped_predictions_v = cropped_predictions_v + b
            corrected_cropped_predictions_alpha = cropped_predictions_alpha + b
            corrected_cropped_predictions_beta = cropped_predictions_beta + b

            loss = evidential_loss(cropped_labels, corrected_cropped_predictions_gamma, corrected_cropped_predictions_v,
                                   corrected_cropped_predictions_alpha, corrected_cropped_predictions_beta,
                                   reduced=False)
            loss = loss * cropped_y_mask
            loss = torch.sum(loss, dim=(1, 2)) / total_pixels_masked

            X.append(loss)
    X = torch.stack(X)
    min_l = torch.min(X, dim=0).values

    return torch.mean(min_l)


def cpsnr(y_true, y_pred, y_mask, size_image):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(2, 3))

            b = torch.reshape(b, (y_shape[0], 1, 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            corrected_mse = torch.sum(
                (cropped_labels_masked - corrected_cropped_predictions) ** 2) / total_pixels_masked

            cPSNR = 10.0 * torch.log10((65535.0 ** 2) / corrected_mse)
            X.append(cPSNR)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)
    return torch.mean(max_cPSNR.values)

def cmse(y_true, y_pred, y_mask, size_image):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(2, 3))

            b = torch.reshape(b, (y_shape[0], 1, 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            corrected_mse = torch.sum(
                (cropped_labels_masked - corrected_cropped_predictions) ** 2) / total_pixels_masked

            X.append(corrected_mse)

    X = torch.stack(X)
    min_cMSE = torch.min(X, dim=0)
    return torch.mean(min_cMSE.values)


def cssim(y_true, y_pred, y_mask, size_image):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(2, 3))

            b = torch.reshape(b, (y_shape[0], 1, 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            cssim = pytorch_ssim.ssim(cropped_labels_masked, corrected_cropped_predictions)

            X.append(cssim)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)
    return torch.mean(max_cPSNR.values)


def cpsnr_returnshift(y_true, y_pred, y_mask, size_image):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    bs = []
    for i in range(max_pixels_shifts + 1):  # range(7)
        for j in range(max_pixels_shifts + 1):  # range(7)
            cropped_labels = y_true[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]
            cropped_y_mask = y_mask[:, :, i:i + (size_image - max_pixels_shifts),
                             j:j + (size_image - max_pixels_shifts)]

            cropped_predictions_masked = cropped_predictions * cropped_y_mask
            cropped_labels_masked = cropped_labels * cropped_y_mask

            total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

            # bias brightness
            b = (1.0 / total_pixels_masked) * torch.sum(cropped_labels_masked - cropped_predictions_masked, dim=(2, 3))

            b = torch.reshape(b, (y_shape[0], 1, 1, 1))

            corrected_cropped_predictions = cropped_predictions_masked + b
            corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

            corrected_mse = torch.sum(
                (cropped_labels_masked - corrected_cropped_predictions) ** 2) / total_pixels_masked

            cPSNR = 10.0 * torch.log10((65535.0 ** 2) / corrected_mse)
            X.append(cPSNR)
            bs.append(b)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)
    pp = torch.argmax(X, dim=0)
    i = pp // (max_pixels_shifts + 1)
    j = pp % (max_pixels_shifts + 1)
    b = bs[pp]

    return torch.mean(max_cPSNR.values), i, j, b


def cpsnr_givenshift(y_true, y_pred, y_mask, size_image, i, j, b):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred[:, :, border:size_image - border, border:size_image - border]

    X = []
    cropped_labels = y_true[:, :, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]
    cropped_y_mask = y_mask[:, :, i:i + (size_image - max_pixels_shifts), j:j + (size_image - max_pixels_shifts)]

    cropped_predictions_masked = cropped_predictions * cropped_y_mask
    cropped_labels_masked = cropped_labels * cropped_y_mask

    total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

    corrected_cropped_predictions = cropped_predictions_masked + b
    corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

    corrected_mse = torch.sum((cropped_labels_masked - corrected_cropped_predictions) ** 2) / total_pixels_masked

    cPSNR = 10.0 * torch.log10((65535.0 ** 2) / corrected_mse)
    X.append(cPSNR)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)
    return torch.mean(max_cPSNR.values)


def cpsnr_nocropshift(y_true, y_pred, y_mask, size_image, b):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = y_true.size()
    border = 3
    max_pixels_shifts = 2 * border
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image * size_croped_image
    cropped_predictions = y_pred

    X = []
    cropped_labels = y_true
    cropped_y_mask = y_mask

    cropped_predictions_masked = cropped_predictions * cropped_y_mask
    cropped_labels_masked = cropped_labels * cropped_y_mask

    total_pixels_masked = torch.sum(cropped_y_mask, dim=(2, 3))

    corrected_cropped_predictions = cropped_predictions_masked + b
    corrected_cropped_predictions = corrected_cropped_predictions * cropped_y_mask

    corrected_mse = torch.sum((cropped_labels_masked - corrected_cropped_predictions) ** 2) / total_pixels_masked

    cPSNR = 10.0 * torch.log10((65535.0 ** 2) / corrected_mse)
    X.append(cPSNR)

    X = torch.stack(X)
    max_cPSNR = torch.max(X, dim=0)
    return torch.mean(max_cPSNR.values)
