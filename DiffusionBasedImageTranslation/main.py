import argparse
from ddpm import Diffusion
from config import Config
from dataloader import get_loader
from torch.backends import cudnn
from utils import *
import wandb

def main(config):
    # For fast training.
    cudnn.benchmark = True

    create_folders(config=config)

    dataloader = get_loader(image_dir=config.image_dir, 
                            selected_attrs=config.selected_attrs,
                            image_size=config.image_size, 
                            batch_size=config.batch_size,
                            num_workers=config.num_workers
                            )
    config_obj = Config(config=config)
    
    diffusion = Diffusion(config=config_obj, dataloader=dataloader)

    if config.mode == 'train':
        print("Training...")
        diffusion.train()
    else:
        print("Testing...")
        diffusion.test()

    if config.wandb:
        wandb.finish()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_in', type=int, default=3, help='dimension of input image')
    parser.add_argument('--c_out', type=int, default=3, help='dimension of output image')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--conv_dim', type=int, default=64, help='number of conv filters in the first layer of the UNet')
    parser.add_argument('--block_depth', type=int, default=3, help='depth of conv layers in encoder/decoder')
    parser.add_argument('--time_emb_dim', type=int, default=256, help='number of channels for time embedding')
    
    # Training configuration.
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes/domains', default=['rain', 'fog', 'night']) # rain, fog, night
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta start for noise scheluder')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta end for noise scheluder')
    parser.add_argument('--s_parameter', type=float, default=0.008, help='Sharpness parameter, controls how "sudden" the change is. A lower value makes the cosine curve smoother.')
    parser.add_argument('--cos_scheduler', type=int, default=0, choices=[0, 1], help='enable cosine scheduler, default is linear scheduler (0 for False, 1 for True)')
    parser.add_argument('--noise_steps', type=int, default=1000, help='noise steps for noise scheluder and sampling')
    parser.add_argument('--model_path', type=str, default=None, help='resume training')
   
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # TODO add val
    parser.add_argument('--wandb', type=int, default=0, choices=[0, 1], help='enable wandb logging (0 for False, 1 for True)')

    # Directories.
    parser.add_argument('--image_dir', type=str, default='DiffusionBasedImageTranslation/data/acdc/rgb_anon')
    parser.add_argument('--log_dir', type=str, default='DiffusionBasedImageTranslation/outputs/logs')
    parser.add_argument('--model_save_dir', type=str, default='DiffusionBasedImageTranslation/outputs/checkpoints')
    parser.add_argument('--sample_dir', type=str, default='DiffusionBasedImageTranslation/outputs/samples')
    parser.add_argument('--result_dir', type=str, default='DiffusionBasedImageTranslation/outputs/results') 

    # Step size.
    parser.add_argument('--log_step', type=int, default=10) # iter
    parser.add_argument('--sample_step', type=int, default=10) # epoch
    parser.add_argument('--validation_step', type=int, default=10) # NOTE validation function not implemented yet
    parser.add_argument('--model_save_step', type=int, default=20) # epoch

    config = parser.parse_args()
    print(config)
    main(config)