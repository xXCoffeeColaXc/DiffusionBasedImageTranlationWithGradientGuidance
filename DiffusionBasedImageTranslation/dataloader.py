
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os

class ACDCDataset(data.Dataset):
    def __init__(self, root_dir, selected_conditions = ['rain', 'fog', 'night'], transform=None):
        self.root_dir = root_dir
        self.selected_conditions = selected_conditions
        self.transform = transform

        self.img_paths = []

        self.preprocess()

    def preprocess(self):
        for condition in self.selected_conditions:
            for split in ['val']:#['train', 'val', 'test']:
                foler_dir = os.path.join(self.root_dir, condition, split)
                for folder in os.listdir(foler_dir):
                    folder_path = os.path.join(foler_dir, folder)
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):
                            self.img_paths.append(os.path.join(folder_path, img_file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image
    

def get_loader(image_dir, selected_attrs, image_size=128, batch_size=16, num_workers=4):
    """Build and return a data loader."""

    mean = torch.tensor([0.4865, 0.4998, 0.4323])
    std = torch.tensor([0.2326, 0.2276, 0.2659])

    # Create Datalaoders
    train_transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # [0, 1] -> [-1, 1]
            transforms.Normalize(mean=mean, std=std), 
        ])
    
    test_transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(image_size),
            transforms.ToTensor(), # [0, 1] -> [-1, 1] 
        ])

    dataset = ACDCDataset(root_dir=image_dir, selected_conditions=selected_attrs, transform=test_transform)
    print(dataset.__len__())

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
