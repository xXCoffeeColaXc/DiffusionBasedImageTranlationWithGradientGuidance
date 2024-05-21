# Diffusion based Image Translation with Semantic mask Guidance

## Project Description
This project focuses on Denoising Diffusion Probabilistic Models (DDPMs) to generate urban scene images in different weather conditions, such as rainy, foggy, night from ideal sunny-day images. For semantic consistency we inject semantic mask as gradient guidance to achive this consistency between the input and target image.

## Repository Files & Functions
- `config.py`: This file contains all configuration settings and parameters required for the project.
- `create_metadata.py`:  Responsible for generating and saving metadata in the form of JSON files. 
- `dataloader.py`: Handles the loading of data for the diffusion model
- `ddpm.py`: This file implements the Denoising Diffusion Probabilistic Model (DDPM). It includes the core functionalities for training and sampling from the diffusion model. The file defines the DDPM class with methods for the forward and reverse diffusion processes, loss computation, and utilities for handling the diffusion steps. It serves as the backbone of the diffusion-based generative model, enabling the generation of new data samples through a trained diffusion process.
- `main.py`: The main entry point of the application. This script handles command-line arguments and orchestrates the initialization and execution of the Diffusion model.
- `modules.py`: This file contains a U-Net implementation which isn't used anymore.
- `ddim_modules.py`: This file houses the architectural modules for the U-Net model used in the diffusion process. It includes definitions for various neural network layers and blocks, such as encoders, decoders, attention mechanisms, and other components specific to the U-Net architecture.
- `metrics.py`: This file contains the implementation of the Kernel Inception Distance metric
- `utils.py`: This file contains utility functions used across the project.
- `visualizer.py`: This file contains tools, to visualize the noising and donising diffusion process.

## Related Works
### Papers:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

### GitHub Repositories:
- [clear-diffusion-keras](https://github.com/beresandras/clear-diffusion-keras)


### Blog Posts:
- [Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/)
- [CUB-200-2011 (Caltech-UCSD Birds-200-2011)](https://paperswithcode.com/dataset/cub-200-2011)
