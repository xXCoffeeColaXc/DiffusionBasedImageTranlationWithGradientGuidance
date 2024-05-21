import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import os

def plot_images(images):
    """
    Plots a batch of images in a single grid.

    Args:
        images (Tensor): A batch of images as a torch tensor.

    Note:
        This function assumes the input tensor is in the format (batch_size, channels, height, width).
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    """
    Saves a batch of images to a specified path.

    Args:
        images (Tensor): A batch of images as a torch tensor.
        path (str): File path to save the image.
        **kwargs: Additional keyword arguments for torchvision.utils.make_grid function.

    Note:
        This function assumes the input tensor is in the format (batch_size, channels, height, width).
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def create_run():
    checkpoint_path = './outputs/checkpoints'
    sample_path = './outputs/samples'
    max_run_id = _find_max_run_id(checkpoint_path)

    new_run_id = max_run_id + 1
    run_dir = f'run_{new_run_id}'
    new_sample_path = os.path.join(sample_path, run_dir)
    new_checkpoint_path = os.path.join(checkpoint_path, run_dir)

    # Create new directories for checkpoints
    os.makedirs(new_sample_path, exist_ok=True)
    os.makedirs(new_checkpoint_path, exist_ok=True)

    print(f"New run directory created: {run_dir}")

    return run_dir

def _find_max_run_id(checkpoint_path):
    max_run_id = 0
    if os.path.exists(checkpoint_path):
        for dir_name in os.listdir(checkpoint_path):
            if dir_name.startswith('run_'):
                try:
                    run_id = int(dir_name.split('_')[1])
                    max_run_id = max(max_run_id, run_id)
                except ValueError:
                    pass  # Ignore directories with non-integer run numbers

    return max_run_id

def create_folders(config):
    """
    Creates directories based on the provided configuration if they do not already exist.

    Args:
        config: Configuration object containing directory paths (log_dir, model_save_dir, sample_dir, result_dir).
    """
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)


