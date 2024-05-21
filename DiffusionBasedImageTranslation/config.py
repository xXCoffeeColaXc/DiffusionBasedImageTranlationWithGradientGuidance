import torch

ROOT_DIR = "./data/CUB_200_2011/CUB_200_2011"
METADATA_DIR = "./data/metadata"


class Config(object):
    def __init__(self, config) -> None:
        # Model configuration.
        self.c_in = config.c_in
        self.c_out = config.c_out
        self.crop_size = config.crop_size
        self.image_size = config.image_size
        self.conv_dim = config.conv_dim
        self.block_depth = config.block_depth
        self.time_emb_dim = config.time_emb_dim

        # Training configurations.
        self.selected_attrs = config.selected_attrs
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.s_parameter = config.s_parameter
        self.cos_scheduler = config.cos_scheduler
        self.noise_steps = config.noise_steps
        self.model_path = config.model_path

        # Miscellaneous.
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.wandb = config.wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.image_dir = config.image_dir
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.sample_dir = config.sample_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validation_step = config.validation_step
        self.model_save_step = config.model_save_step

