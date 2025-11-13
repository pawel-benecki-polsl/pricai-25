import lpips
import torch
import os


class Models:
    __curr_dir = os.path.dirname(os.path.abspath(__file__))

    LPIPS_KFS_20221206 = os.path.join(__curr_dir, "lpips_trained_on_kfs_sift_20221206.pth")
    LPIPS_1VOTE_KFS_2VOTES_HUMAN = os.path.join(__curr_dir, "lpips_trained_1vote_kfs_2votes_human.pth")


def load_lpips(path: str = None):
    if path is None:
        print("Loading pure LPIPS")
        return lpips.LPIPS().cuda().eval().requires_grad_(False)
    else:
        print(f"Loading LPIPS with weights from {path}")
        result = lpips.LPIPS().cuda()
        loaded = torch.load(path)
        problems = result.load_state_dict(loaded)
        return result.eval().requires_grad_(False)
