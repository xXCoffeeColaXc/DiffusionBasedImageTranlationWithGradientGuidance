import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from ddim_modules import UNet
from torchvision.transforms import transforms
from tqdm import tqdm
import os

class DiffusionVisualizer(object):
    def __init__(self, cfg, image_path, model_checkpoint_path):
        self.config = cfg
        self.image_path = image_path
        self.model_checkpoint_path = model_checkpoint_path
        self.build_model()
        self.preprocess_image()

    def build_model(self):
        # Create
        self.unet = UNet(self.config['c_in'], self.config['c_out'], self.config['image_size'], self.config['conv_dim'], self.config['block_depth'], self.config['time_emb_dim'])
        
        # Load
        checkpoint = torch.load(self.model_checkpoint_path, map_location=lambda storage, loc: storage)
        self.unet.load_state_dict(checkpoint['model_state_dict'])

        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(self.config['device'])
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.unet.to(self.config['device'])
        

    def preprocess_image(self):
        self.mean = torch.tensor([0.4865, 0.4998, 0.4323])
        self.std = torch.tensor([0.2326, 0.2276, 0.2659])
        
        base_transforms = [
            transforms.Resize(self.config['image_size'], transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(self.config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        transform = transforms.Compose(base_transforms)

        image = Image.open(self.image_path).convert('RGB')
 
        self.test_image = transform(image)
        self.test_image = self.test_image.to(self.config['device'])
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.config['beta_start'], self.config['beta_end'], self.config['noise_steps'])
    
    def forward_process(self, x, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        # Remove the batch dimension if it was originally a single image
        if noisy_image.size(0) == 1:
            noisy_image = noisy_image.squeeze(0)
        
        return noisy_image
    
    def add_noise_for_steps(self, num_steps=10):
        # Ensure test_image is a single image (not batched)
        test_image_single = self.test_image.squeeze(0)  # Removes the batch dimension if it's present
        noisy_images = [test_image_single.cpu().detach()]

        for step in range(1, num_steps):
            # Gradually increase the noise level
            noise_level = step / num_steps
            t = torch.tensor([int(self.config['noise_steps'] * noise_level)]).to(self.config['device'])

            noise = torch.randn_like(test_image_single).to(self.config['device'])
            noisy_image = self.forward_process(test_image_single, noise, t)
            noisy_images.append(noisy_image.cpu().detach())

        # Add pure noise in the last step
        pure_noise = torch.randn_like(test_image_single).to(self.config['device'])
        noisy_images.append(pure_noise.cpu().detach())

        return noisy_images


    def remove_noise_for_steps(self, noise_image, num_steps=10):
        # Start with pure noise
        #noise_image = torch.randn((1, 3, self.config['image_size'], self.config['image_size'])).to(self.config['device'])
        noise_image = noise_image.unsqueeze(0).to(self.config['device'])
        denoised_images = [noise_image.squeeze(0).cpu().detach()]

        step_interval = self.config['noise_steps'] // num_steps  # Calculate the interval for saving images

        for step in reversed(range(1, self.config['noise_steps'])):
            self.unet.eval()
            with torch.no_grad():
                t = torch.tensor([step]).to(self.config['device'])
                one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None]
                predicted_noise = self.unet(noise_image, one_minus_alpha_hat) if self.unet.requires_alpha_hat_timestep else self.unet(noise_image, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(noise_image) if step < self.config['noise_steps'] - 1 else torch.zeros_like(noise_image)
                noise_image = 1 / torch.sqrt(alpha) * (noise_image - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

                # Save and visualize every 100th step
                if step % step_interval == 0 or step == 1:
                    denoised_images.append(noise_image.squeeze(0).cpu().detach())

        return denoised_images

    
    def denorm(self, image):
        mean_expanded = self.mean.view(3, 1, 1).cpu().detach()
        std_expanded = self.std.view(3, 1, 1).cpu().detach()

        # Denormalize
        x_adj = (image * std_expanded + mean_expanded) * 255
        x_adj = x_adj.clamp(0, 255).type(torch.uint8)
        return x_adj

    def visualize(self, images, title):
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        for i, image in enumerate(images):
            ax = axes[i]
            image = self.denorm(image)
            ax.imshow(image.permute(1, 2, 0))  # Adjust for PyTorch channel order
            ax.axis('off')
        plt.suptitle(title)
        plt.show()

    def visualize_translation(self, source_image, target_image):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        source_image = self.denorm(source_image)
        axes[0].imshow(source_image.permute(1, 2, 0))
        axes[0].set_title('Source Image')
        axes[0].axis('off')
        target_image = self.denorm(target_image)
        axes[1].imshow(target_image.permute(1, 2, 0))
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        plt.show()

    

if __name__ == '__main__':
    cfg = {
        'c_in': 3,
        'c_out': 3,
        'image_size': 128,
        'conv_dim': 64,
        'block_depth': 3,
        'time_emb_dim': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'noise_steps': 1000
    }
    test_image_path = 'data/leftImg8bit_trainvaltest/leftImg8bit/train/zurich/zurich_000038_000019_leftImg8bit.png'
    test_images = os.listdir('data/leftImg8bit_trainvaltest/leftImg8bit/test/leverkusen')
    model_path = 'outputs/checkpoints/run_12/500-checkpoint.ckpt'
    visualizer = DiffusionVisualizer(cfg, test_image_path, model_path)
    num_steps = 1000

    batch_size = 4
    random_batch = np.random.choice(len(test_images), size=batch_size, replace=False)
    print(random_batch)
    test_batch = [test_images[i] for i in random_batch]

    noisy_images = visualizer.add_noise_for_steps(num_steps)
    #visualizer.visualize(noisy_images, 'Noisy images')
    print(noisy_images[-1].shape)
    denoised_images = visualizer.remove_noise_for_steps(noisy_images[-1], num_steps)
    #visualizer.visualize(denoised_images, 'Denoised images')
 
    visualizer.visualize_translation(noisy_images[0], denoised_images[-1])
