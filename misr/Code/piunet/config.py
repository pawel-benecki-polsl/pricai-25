import os


class Config(object):
    BASE_DIR = "C:\\projekty\\piunet\\Dataset\\"

    def __init__(self):
        # NIR
        self.train_lr_file = os.path.join(Config.BASE_DIR, "X_NIR_train.npy")
        self.train_hr_file = os.path.join(Config.BASE_DIR, "y_NIR_train.npy")
        self.train_masks_file = os.path.join(Config.BASE_DIR, "y_NIR_train_masks.npy")

        self.val_lr_file = os.path.join(Config.BASE_DIR, "X_RED_val.npy")
        self.val_hr_file = os.path.join(Config.BASE_DIR, "y_RED_val.npy")
        self.val_masks_file = os.path.join(Config.BASE_DIR, "y_RED_val_masks.npy")

        # self.val_lr_file = os.path.join(Config.BASE_DIR, "X_NIR_val.npy")
        # self.val_hr_file = os.path.join(Config.BASE_DIR, "y_NIR_val.npy")
        # self.val_masks_file = os.path.join(Config.BASE_DIR, "y_NIR_val_masks.npy")

        self.max_train_scenes = 50
        self.max_val_scenes = 10
        # self.max_train_scenes = 393
        # RED

        self.device = "cuda"
        self.validate = True

        # architecture
        self.N_feat = 42
        self.R_bneck = 8
        self.N_tefa = 16
        self.N_heads = 1
        self.patch_size = 32

        # learning
        self.batch_size = 4  # 8
        self.N_epoch = 750
        self.learning_rate = 1e-4
        self.workers = 2  # 5

        # logging
        self.log_every_iter = 20
        self.validate_every_iter = 500
        self.save_every_iter = 500


class ConfigRED(Config):
    BASE_DIR = "C:\\projekty\\piunet\\Dataset\\"

    def __init__(self):
        super().__init__()
        # NIR
        self.train_lr_file = os.path.join(Config.BASE_DIR, "X_RED_train.npy")
        self.train_hr_file = os.path.join(Config.BASE_DIR, "y_RED_train.npy")
        self.train_masks_file = os.path.join(Config.BASE_DIR, "y_RED_train_masks.npy")

        self.val_lr_file = os.path.join(Config.BASE_DIR, "X_RED_val.npy")
        self.val_hr_file = os.path.join(Config.BASE_DIR, "y_RED_val.npy")
        self.val_masks_file = os.path.join(Config.BASE_DIR, "y_RED_val_masks.npy")


class ConfigNIR(Config):
    BASE_DIR = "C:\\projekty\\piunet\\Dataset\\"

    def __init__(self):
        super().__init__()
        # NIR
        self.train_lr_file = os.path.join(Config.BASE_DIR, "X_NIR_train.npy")
        self.train_hr_file = os.path.join(Config.BASE_DIR, "y_NIR_train.npy")
        self.train_masks_file = os.path.join(Config.BASE_DIR, "y_NIR_train_masks.npy")

        self.val_lr_file = os.path.join(Config.BASE_DIR, "X_NIR_val.npy")
        self.val_hr_file = os.path.join(Config.BASE_DIR, "y_NIR_val.npy")
        self.val_masks_file = os.path.join(Config.BASE_DIR, "y_NIR_val_masks.npy")
