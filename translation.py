import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import Rein.rein
import torch.nn.functional as F
from tqdm import tqdm
from DiffusionBasedImageTranslation.ddim_modules import UNet
import time
from DiffusionBasedImageTranslation.seg_modules import MyCityscapesDataset, transform, transform_mask
from torch.utils.data import DataLoader
import math
import os
# import segmentation_models_pytorch as smp
from mmengine.config import Config
from mmseg.apis.inference import inference_model,show_result_pyplot,init_model
import matplotlib.pyplot as plt
import torch.nn as nn
from mmseg.apis.utils import _preprare_data
import random

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



class DiffusionTranslationSGG(object):
    def __init__(self, cfg, model_checkpoint_path, seg_config_path, seg_model_path):
        self.config = cfg
        self.build_model(model_checkpoint_path)
        self.build_seg_model(seg_config_path, seg_model_path)

    def build_model(self, model_checkpoint_path):
        # Create
        self.unet = UNet(self.config['c_in'], self.config['c_out'], self.config['image_size'], self.config['conv_dim'], self.config['block_depth'], self.config['time_emb_dim'])
    
        # Load
        checkpoint = torch.load(model_checkpoint_path, map_location=lambda storage, loc: storage)
        self.unet.load_state_dict(checkpoint['model_state_dict'])

        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(self.config['device'])
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.unet.to(self.config['device'])
        
    def build_seg_model(self, seg_config_path, seg_model_path):
        cfg=Config.fromfile(seg_config_path)
        self.seg_model:nn.Module=init_model(cfg,seg_model_path,'cuda:0')
        self.seg_model.cfg=cfg
        self.seg_model.cfg.test_pipeline = self.seg_model.cfg.cityscapes_test_pipeline
       
        num_classes = 19
        class_weights = torch.ones(num_classes)

        # Set higher weights for important classes
        important_classes = [0, 1, 2, 8, 10, 13]
        for idx in important_classes:
            class_weights[idx] = 10.0  # Or any other higher value

        self.class_weights = class_weights.to(self.config['device'])
        print("Class Weights:", class_weights)

    def ddpm_sample(self, n):
        """Performs sampling of images using DDPM method.

        Args:
            n: Number of images to sample.

        Returns:
            Torch tensor of sampled images.
        """
        print(f"Sampling {n} new images...")
        self.unet.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.config["image_size"], self.config["image_size"])).to(self.config["device"])
            for i in reversed(range(1, self.config["noise_steps"])):
                t = (torch.ones(n) * i).long().to(self.config["device"]) # create a tensor of lenght n with the current timestep
                one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None] # None = add new dimension
                if self.unet.requires_alpha_hat_timestep:
                    predicted_noise = self.unet(x, one_minus_alpha_hat)
                else:
                    predicted_noise = self.unet(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # x(t) = 1/sqrt(alpha)*[x - (1-alpha)/(sqrt(1-alpha_hat))*predicted_noise] + sqrt(beta)*noise
                # [x - (1-alpha)/(sqrt(1-alpha_hat))*predicted_noise] = denoising operation, substract the sclaed predicted noise from the image
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        self.unet.train()
        return x.cpu().detach()
        
    def denorm(self, data, mean=(0.4865, 0.4998, 0.4323), std=(0.2326, 0.2276, 0.2659)):
        
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(data.device)
            std = torch.tensor(std).reshape(1, -1, 1, 1).to(data.device)
            if len(data.shape) == 3:  # If there's no batch dimension, add it
                data = data.unsqueeze(0)
            denormalized = data * std + mean
            denormalized = torch.clamp(denormalized * 255, 0, 255).byte() 
            return denormalized.squeeze(0) if denormalized.shape[0] == 1 else denormalized

        elif isinstance(data, np.ndarray):
            mean = np.array(mean).reshape(-1, 1, 1)
            std = np.array(std).reshape(-1, 1, 1)
            if data.ndim == 3:  # If there's no batch dimension, expand dimensions
                data = np.expand_dims(data, 0)
            denormalized = data * std + mean
            denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
            return denormalized.squeeze(0) if denormalized.shape[0] == 1 else denormalized

        else:
            raise TypeError("Input must be a numpy array or a torch tensor")
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.config['beta_start'], self.config['beta_end'], self.config['noise_steps'])
    
    def adjust_noise_schedule(self, remaining_steps):
        # Adjust the noise schedule for the remaining steps
        self.beta = self.prepare_noise_schedule()[:remaining_steps].to(self.config['device'])
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def forward_process(self, x, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        # Remove the batch dimension if it was originally a single image
        if noisy_image.size(0) == 1:
            noisy_image = noisy_image.squeeze(0)
        
        return noisy_image
    
    def add_noise_for_steps(self, diff_input_image):
        # Ensure test_image is a single image (not batched)
        test_image_single = diff_input_image
        noisy_images = [test_image_single.cpu().detach()]

        for step in range(1, self.config['num_steps']):
            # Gradually increase the noise level
            noise_level = step / self.config['num_steps']
            t = torch.tensor([int(self.config['noise_steps'] * noise_level)]).to(self.config['device'])

            noise = torch.randn_like(test_image_single).to(self.config['device'])
            noisy_image = self.forward_process(test_image_single, noise, t)
            noisy_images.append(noisy_image.cpu().detach())
        
        return noisy_images # NOTE: normed tensors on cpu

    def remove_noise_for_steps(self, noise_image, gt, seg_input_image, lambda_lcg=60.0, lambda_gsg=60.0, remaining_steps=500):
        # Start with pure noise
        #noise_image = torch.randn((1, 3, self.config['image_size'], self.config['image_size'])).to(self.config['device'])
        x_t = noise_image.unsqueeze(0).to(self.config['device']) # NOTE: normed tensor on device   
        denoised_images = [noise_image.squeeze(0).cpu().detach()]
        decoded_outputs = []
        #gt = torch.tensor(self.seg_input_mask, dtype=torch.long).unsqueeze(0).to(self.config['device']) # NOTE :not normed tensor on device # [1, 128, 128]
        
        # lambda_gsg = 60.0
        # lambda_lcg = 60.0

        noise_step = self.config['noise_steps'] - (remaining_steps + 100)
        vis_img = None
        self.step_interval = noise_step // (self.config['num_steps'])  # Calculate the interval for saving images
       
        with tqdm(total=noise_step, desc='Reverse Process', unit='step') as pbar:
            for step in reversed(range(1, noise_step)):
                self.unet.eval()
                with torch.no_grad():
                    t = torch.tensor([step]).to(self.config['device'])
                    one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None]
                    
                    # Predict the noise for the current timestep
                    predicted_noise = self.unet(x_t, one_minus_alpha_hat) if self.unet.requires_alpha_hat_timestep else self.unet(noise_image, t)
                    
                    # Calculate the alpha, alpha_hat, beta for the current timestep
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    noise = torch.randn_like(x_t) if step > 1 else torch.zeros_like(x_t) 
                    noise = noise.to(self.config['device'])
                    cov = torch.sqrt(beta) * noise

                    # Calculate mu for the current timestep
                    mu = 1 / torch.sqrt(alpha) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
                    
                    # Apply SGG
                    if step % 2 == 0:
                        # Apply LCG
                        x_t_minus_1 = self.apply_lcg(x_t, mu, gt, lambda_lcg, cov)
                        #x_t_minus_1 = mu + cov
                        
                    elif step % 2 == 1:
                        # Apply GSG
                        x_t_minus_1, vis_img = self.apply_gsg(x_t, mu, gt, lambda_gsg, cov, seg_input_image)
                        #x_t_minus_1 = mu + cov
                    
                    x_t = x_t_minus_1.clone() # update x_t for the next step
                    
                    # Save and visualize every 100th step
                    if step % self.step_interval == 0 or step == 1:
                        denoised_images.append(x_t.squeeze(0).cpu().detach())
                        if vis_img is not None: 
                            decoded_outputs.append(vis_img)

                    # Update tqdm progress bar with relevant information
                    pbar.set_postfix({"Current Step": step, "Guidance": "LCG" if step % 2 == 0 else "GSG"})
                    pbar.update(1)

        return denoised_images, decoded_outputs
    
    def apply_gsg(self, x_t, mu, y, lambda_val, cov, seg_input_image):
        # Adjust mu based on global guidance
        # L(global)[x, y] = L(ce)[g(x), y], 
        # mu_hat(x,k) = mu(x,k) + lambda * cov * gradient(L(global)[x,y])
        
        self.seg_model.eval()

        for param in self.seg_model.parameters():
            param.requires_grad = False

        with torch.set_grad_enabled(True):

            # Format x_t to numpy array for segmentation model
            x_t_dn = self.denorm(x_t.clone())  # denormed [1, 3, 128, 128]
            x_t_dn = x_t_dn.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [128, 128, 3]
            
            # prepare data for segmentation model
            data, _ = _preprare_data(x_t_dn, self.seg_model)
            data['inputs'][0] = data['inputs'][0].float() 
            data['inputs'][0].requires_grad_(True)

            # predict segmentation mask
            results = self.seg_model.test_step(data)
            
            if not results[0].seg_logits.data.requires_grad:
                raise RuntimeError("Model output does not require gradients, check model configuration.")

            logits = results[0].seg_logits.data.unsqueeze(0) # (1, 19, 128, 128)
            
            # Calculate cross-entropy loss
            ignore_index = 19
            loss = F.cross_entropy(logits, y, weight=self.class_weights, ignore_index=ignore_index)
            #print("[GSG] Cross-entropy Loss:", loss.item())

            # If loss is zero (no overlap), we cannot backpropagate
            if loss.item() == 0:
                raise ValueError('Cross Entropy loss is zero, gradient cannot be computed.')
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require gradients, ensure inputs and model are correctly configured.")

            loss.backward()
            grads = data['inputs'][0].grad.unsqueeze(0).to(self.config['device'])
            # grads = torch.autograd.grad(outputs=loss, inputs=data['inputs'][0], only_inputs=True)[0]
            # grads = grads.unsqueeze(0).to(self.config['device'])

            # grads = grads * 255.0 # Scale the gradients to the [0, 255] range

            clip_value = 1.0  # Adjust clip value as needed, considering [0, 255] range
            grads = torch.clamp(grads, -clip_value, clip_value)
            
            with torch.no_grad():
                mu_hat = mu - lambda_val * cov * grads
                mu_hat = mu_hat.detach()
                data['inputs'][0].grad.zero_()
        
        vis_img = show_result_pyplot(model=self.seg_model, img=seg_input_image, result=results[0], with_labels=False, opacity=1.0, show=False)
    
        self.seg_model.train()

        x_t_minus_1 = mu_hat + cov # sampling x_t from adjusted mu with added noise

        return x_t_minus_1, vis_img
    
    def apply_lcg(self, x_t, mu, y, lambda_val, cov):
        device = self.config['device']
        x_t_minus_1 = torch.zeros_like(mu).to(device)
        num_classes = 19

        self.seg_model.eval()
        for param in self.seg_model.parameters():
            param.requires_grad = False

        with torch.set_grad_enabled(True):
            # Denormalize input image
            x_k = self.denorm(x_t.clone())  # denormed [1, 3, 128, 128]
                
            for c in range(num_classes):
                # Generate class-specific mask
                mc = (y == c).long().unsqueeze(1).to(device) # [1,1,128,128]
            
                if len(mc.unique()) > 1:
                    x_k_masked = x_k * mc # [1,3,128,128]

                    #print("Selected Class:", c)
                    if c == 0:
                        y_c = torch.logical_not(mc.squeeze(0)).long() # swap 0 and 1 pixels [1,128,128]
                    else:
                        y_c = y * mc.squeeze(0) # [1,128,128]

                    # Format x_t to numpy array for segmentation model
                    x_k_masked = x_k_masked.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  # [128, 128, 3]

                    # prepare data for segmentation model
                    data, _ = _preprare_data(x_k_masked, self.seg_model)
                    data['inputs'][0] = data['inputs'][0].float()
                    data['inputs'][0].requires_grad_(True)

                    
                    # predict segmentation mask
                    results = self.seg_model.test_step(data)

                    if not results[0].seg_logits.data.requires_grad:
                        raise RuntimeError("Model output does not require gradients, check model configuration.")

                    logits = results[0].seg_logits.data.unsqueeze(0) # (1, 19, 128, 128)
                    
                    # Calculate cross-entropy loss
                    if c == 0: # the index of road class is 0, which in the binary mask is the background
                        ignore_index = 1 # so we flip the two classes
                        loss = F.cross_entropy(logits, y_c, ignore_index=ignore_index)
                    else:
                        ignore_index = 0 # background class
                        loss = F.cross_entropy(logits, y_c, ignore_index=ignore_index)
                   
                    #print("[LCG] Cross-entropy Loss:", loss.item())

                    # If loss is zero (no overlap), we cannot backpropagate
                    if loss.item() == 0:
                        raise ValueError('Cross Entropy loss is zero, gradient cannot be computed.')
                    if not loss.requires_grad:
                        raise RuntimeError("Loss does not require gradients, ensure inputs and model are correctly configured.")

                    loss.backward()
                    #print("LCG grads after: ", data['inputs'][0].grad.unique())
                    
                    grads = data['inputs'][0].grad.unsqueeze(0).to(device)
                    
                    # grads = grads * 255.0 # Scale the gradients to the [0, 255] range
                    # print("LCG scaled grads: ", grads.min(), grads.max())

                    clip_value = 1.0  # Adjust clip value as needed, considering [0, 255] range
                    grads = torch.clamp(grads, -clip_value, clip_value)


                    
                    #grads = torch.autograd.grad(outputs=loss, inputs=data['inputs'][0], only_inputs=True)[0]
                    #print(grads.norm())
                    #grads = torch.clamp(grads, min=-1.0, max=1.0)
                    #grads = grads.unsqueeze(0).to(self.config['device'])

                    # Scale the gradients to the [0, 255] range
                    #grads = grads * 255.0
                    #print("GSG scaled grads: ", grads.min(), grads.max())

                    #clip_value = 1.0  # Adjust clip value as needed, considering [0, 255] range
                    #grads = torch.clamp(grads, -clip_value, clip_value)
                    #grads = grads * mc
                    
                    with torch.no_grad():
                        mu_hat = mu - lambda_val * cov * grads

                        data['inputs'][0].grad.zero_()

                        mu_hat = mu_hat.detach()

                        x_t_minus_1_c = mu_hat + cov
                        x_t_minus_1 += x_t_minus_1_c * mc

                    # plt.imshow(self.denorm(x_t_minus_1).squeeze(0).cpu().detach().permute(1, 2, 0).numpy())
                    # plt.show()

        if 19 in y.unique():
            # Generate mask for the void class
            mc = (y == 19).long().unsqueeze(1).to(device) # [1,1,128,128]
            x_t_minus_1 += (mu + cov) * mc
                    

        self.seg_model.train()
            
        return x_t_minus_1
    
    def save_images(self, images, save_path, denorm=True):
        for i, image in enumerate(images):
            if denorm:
                image = self.denorm(image)
                image = image.permute(1, 2, 0).numpy()
            plt.imsave(f"{save_path}_{i}.png", image)

    def visualize(self, images, title, denorm=True):
        num_images = len(images)
        num_cols = 4
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        axes = axes.flatten()  # Ensure consistent indexing across multiple rows

        for i, image in enumerate(images):
            ax = axes[i]
            if denorm:
                image = self.denorm(image)
                ax.imshow(image.permute(1, 2, 0))
            else:
                ax.imshow(image)
            ax.axis('off')
        # Hide any unused subplots in the grid
        for j in range(i + 1, num_rows * num_cols):
            axes[j].axis('off')
        plt.suptitle(title)
        plt.show()

    def visualize_translation(self, source_image, target_image, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        source_image = self.denorm(source_image)
        axes[0].imshow(source_image.permute(1, 2, 0))
        axes[0].set_title('Source Image')
        axes[0].axis('off')
        target_image = self.denorm(target_image)
        axes[1].imshow(target_image.permute(1, 2, 0))
        axes[1].set_title('Target Image')
        axes[1].axis('off')
        if save_path:
            plt.savefig(save_path) 
        #plt.show()


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
        'noise_steps': 1000, # controlls how many steps
        'num_steps': 10, # controlls how many steps to visualize
    }

    model_path = '/home/talmacsi/Documents/BME/Onlab2/AWS_outputs/run_11/1000-checkpoint.ckpt'
    seg_model_path = 'Rein/checkpoints/dinov2_segmentor_acdc.pth'
    seg_config_path = 'Rein/configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py'

    test_dataset = MyCityscapesDataset('DiffusionBasedImageTranslation/data/cityscapes', split='val', mode='fine',
                       target_type='semantic',transform=transform, target_transform=transform_mask)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # sample = 25 #
    # image_tensor, image_numpy, mask = next(iter(test_loader))
    
    translator = DiffusionTranslationSGG(cfg=cfg, 
                                         model_checkpoint_path=model_path, 
                                         seg_config_path=seg_config_path,
                                         seg_model_path=seg_model_path
                                         )

    
    
    #sampled_images = translator.ddpm_sample(64)
    #translator.save_images(sampled_images, 'outputs/samples/run_1', denorm=True)
    #translator.visualize(sampled_images, 'Sampled Images', True)
    lambda_gsg = 60.0
    lamda_lcg = 60.0
    remaining_step = 600
    # seed = 428400717

    base_path = 'results/all_res'
    save_path = f'{base_path}/lcg_{lamda_lcg}_gsg_{lambda_gsg}_N_{remaining_step}_{time.time()}'
    os.makedirs(save_path, exist_ok=True)

    for idx, (batch_image_tensor, batch_image_numpy, batch_mask) in enumerate(test_loader):
        for i in range(test_loader.batch_size):
            os.makedirs(f'{save_path}/{idx}/{i}', exist_ok=True)
            random_seed = random.randint(1, 2**32 - 1)
            print("Random Seed:", random_seed)
            set_seeds(random_seed)

            gt = batch_mask[i].long().unsqueeze(0).to(cfg['device'])
            diff_input_image = batch_image_tensor[i].to(cfg['device'])
            seg_input_image = np.array(batch_image_numpy[i])

            translator.adjust_noise_schedule(cfg['noise_steps'])
            noisy_images = translator.add_noise_for_steps(diff_input_image) 
            #translator.visualize(noisy_images, 'Noisy Images', True)
            translator.adjust_noise_schedule(remaining_step)

            denoised_images, decoded_outputs = translator.remove_noise_for_steps(noise_image=noisy_images[4], 
                                                                                 gt=gt, 
                                                                                 seg_input_image=seg_input_image,
                                                                                 lambda_gsg=lambda_gsg, 
                                                                                 lambda_lcg=lamda_lcg, 
                                                                                 remaining_steps=remaining_step)
            #translator.visualize(denoised_images, 'Denoised Images', True)
            denoised_images_save_path = f'{save_path}/{idx}/{i}/denoised_images.png'
            translator.save_images(denoised_images, denoised_images_save_path, denorm=True)
            
            if decoded_outputs:
                #translator.visualize(decoded_outputs, 'Mask Images', False)
                # Save the last decoded output image
                #last_decoded_output = decoded_outputs[-1] 
                mask_save_path = f'{save_path}/{idx}/{i}/masks.png'
                #plt.imsave(mask_save_path, last_decoded_output)
                translator.save_images(decoded_outputs, mask_save_path, denorm=False)

            # Save the last denoised image
            #last_denoised_image = denoised_images[-1]
            #last_denoised_image_save_path = f'{save_path}/last_denoised_image_{idx}_{i}.png'
            # last_denoised_image = translator.denorm(last_denoised_image)
            # last_denoised_image = last_denoised_image.permute(1, 2, 0).numpy()
            # plt.imsave(last_denoised_image_save_path, last_denoised_image)
            #translator.save_images([last_denoised_image], last_denoised_image_save_path, denorm=True)

            
            # Visualize the translation from the first noisy image to the last denoised image
            transaltion_save_path = f'{save_path}/{idx}/{i}/translation.png'
            translator.visualize_translation(noisy_images[0], denoised_images[-1], save_path=transaltion_save_path)
