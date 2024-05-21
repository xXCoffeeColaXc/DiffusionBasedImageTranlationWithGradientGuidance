import torch
# import torchmetrics
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# from pytorch_lightning import seed_everything, LightningModule, Trainer
# from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
# import segmentation_models_pytorch as smp
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle']
#why i choose 20 classes
#https://stackoverflow.com/a/64242989

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)

colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))
mean = torch.tensor([0.4865, 0.4998, 0.4323])
std = torch.tensor([0.2326, 0.2276, 0.2659])
image_size = 128

transform = transforms.Compose([
    transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),  # Resize the smallest side to 128 and maintain aspect ratio
    transforms.CenterCrop(image_size), 
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # [0, 1] -> [-1, 1]
    transforms.Normalize(mean=mean, std=std), 
])

transform_seg_image = transforms.Compose([
    transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),  # Use NEAREST for masks to avoid interpolation of label classes
    transforms.CenterCrop(image_size),
    #transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize(image_size, transforms.InterpolationMode.NEAREST),  # Use NEAREST for masks to avoid interpolation of label classes
    transforms.CenterCrop(image_size),
    #transforms.ToTensor(),
])


def encode_segmap(mask):
    #remove unwanted classes and recitify the labels of wanted classes
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    temp = np.squeeze(temp)
    
    # Initialize the RGB image
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)

    # Populate the RGB values
    for l in range(0, n_classes):
        mask = (temp == l)
        rgb[mask, 0] = label_colours[l][0] / 255.0
        rgb[mask, 1] = label_colours[l][1] / 255.0
        rgb[mask, 2] = label_colours[l][2] / 255.0

    return rgb

from .labels import labels
class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]
ignore_index = 19
map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1
        if label.hasInstances is True:
            inst_map_to_id[label.id] = j
            j += 1

id_to_map = {id: i for i, id in map_to_id.items()}
inst_id_to_map = {id: i for i, id in inst_map_to_id.items()}

class MyCityscapesDataset(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if image is None:
            print(self.images[index])
            dummy_image = torch.randn(3, 256, 256) 
            target = self.target_transform(target) if isinstance(target, Image.Image) else target
            target = np.array(target)
            return dummy_image, dummy_image, target
            

        if self.transforms is not None:
            image_tensor = self.transform(image)
            image_numpy = transform_seg_image(image)
            image_numpy = np.array(image_numpy)
            target = self.target_transform(target) if isinstance(target, Image.Image) else target
            target = np.array(target)
            
            # Map the pixel values using the dictionary
            mapped_mask = np.vectorize(map_to_id.get)(target, ignore_index)  # Use -1 for unmapped pixels
                
        return image_tensor, image_numpy, mapped_mask


# class SegmentorModel(LightningModule):
#     def __init__(self, backbone_name='resnet34'):
#         super(SegmentorModel,self).__init__()

#         #architecute
#         self.layer = smp.Unet(
#                     encoder_name=backbone_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#                     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#                     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#                     classes=n_classes,                      # model output channels (number of classes in your dataset)
#                 ).train()
  
#         #parameters
#         self.lr=1e-3
#         self.batch_size=128
#         self.numworker=multiprocessing.cpu_count()//4

#         self.criterion= smp.losses.DiceLoss(mode='multiclass')
#         self.metrics = torchmetrics.JaccardIndex(num_classes=n_classes, task='multiclass')
        
#         self.train_class = MyCityscapesDataset('data/cityscapes', split='train', mode='fine',
#                         target_type='semantic',transform=transform, target_transform=transform_mask)
#         self.val_class = MyCityscapesDataset('data/cityscapes', split='val', mode='fine',
#                         target_type='semantic',transform=transform, target_transform=transform_mask)
        
        
#     def process(self,image,segment):
#         out=self(image)
#         segment=encode_segmap(segment)
#         loss=self.criterion(out,segment.long())
#         iou=self.metrics(out,segment)
#         return loss,iou
        
#     def forward(self,x):
#         return self.layer(x)


#     def configure_optimizers(self):
#         opt=torch.optim.AdamW(self.parameters(), lr=self.lr)
#         return opt

#     def train_dataloader(self):
#         return DataLoader(self.train_class, batch_size=self.batch_size, 
#                         shuffle=True,num_workers=self.numworker,pin_memory=True)

#     def training_step(self,batch,batch_idx):
#         image,segment=batch
#         loss,iou=self.process(image,segment)
#         self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
#         self.log('train_iou', iou,on_step=False, on_epoch=True,prog_bar=False)
#         return loss

#     def val_dataloader(self):
#         return DataLoader(self.val_class, batch_size=self.batch_size, 
#                         shuffle=False,num_workers=self.numworker,pin_memory=True)
        
#     def validation_step(self,batch,batch_idx):
#         image,segment=batch
#         loss,iou=self.process(image,segment)
#         self.log('val_loss', loss,on_step=False, on_epoch=True,prog_bar=False)
#         self.log('val_iou', iou,on_step=False, on_epoch=True,prog_bar=False)
#         return loss
    

# def train():
#     model = SegmentorModel()
#     checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints',
#                                     filename='file',save_last=True)
#     trainer = Trainer(max_epochs=200,
#                 devices=-1,
#                 precision=16,
#                 callbacks=[checkpoint_callback],
#                 )
    
#     trainer.fit(model)

# # -------------------------------------------------------------------
# # -------------------------- WRAPPER CLASS --------------------------
# # -------------------------------------------------------------------

# class Segmentor:
#     def __init__(self, model_weight_path, device, backbone_name='resnet34') -> None:
#         self.device = device
#         self.build_load_model(backbone_name, model_weight_path)
    
#     # TODO: Add method to build model, seperate function to load model from pth file
#     def build_load_model(self, backbone_name, model_weight_path):

#         self.model = SegmentorModel(backbone_name)
#         self.model.load_state_dict(torch.load(model_weight_path))
#         #self.model.eval()
        
#         self.model.to(self.device)

#     # TODO: Add method to load model from checkpoint
#     def load_from_checkpoint(self, model_weight_path):
#         self.model.load_state_dict(torch.load(model_weight_path))
#         self.model.eval()
#         self.model.to(self.device)
        
#     def predict(self, img):
        
#         if len(img.shape) == 3:
#             img = img.unsqueeze(0)  # add batch dimension
#         #img = img.to(self.device)
#         #print("image shape:", img.shape)  # torch.Size([1, 3, 128, 128])
#         with torch.set_grad_enabled(True):
#             output = self.model(img)
#         #output = self.model(img)  # forward pass
#         #print("pred shape:", output.shape)  # torch.Size([1, 20, 128, 128])
#         #output = torch.argmax(output, dim=0) # get the class index with highest probability
#         return output
    
#     def decode_pred_mask(self, output):
#         outputx=output.detach().cpu()[0]
#         decoded_ouput=decode_segmap(torch.argmax(outputx,0))
#         return decoded_ouput


#     def preprocess_mask(self, mask):
#         '''
#         This method only need if you want to visualize the already preprocessed mask
#         '''
#         encoded_mask=encode_segmap(mask.clone()) 
#         decoded_mask=decode_segmap(encoded_mask.clone()) 
#         return decoded_mask

#     def denorm(image):

#         mean = torch.tensor([0.4865, 0.4998, 0.4323])
#         std = torch.tensor([0.2326, 0.2276, 0.2659])

#         # Denormalize
#         x_adj = (image * std + mean) * 255
#         x_adj = x_adj.clamp(0, 255).type(torch.uint8)
#         return x_adj
    

    
# if __name__ == '__main__':

#     # TRAIN MODEL
#     #train()

#     # RUN INFERENCE ON SINGLE IMAGE WITH WRAPPER CLASS
#     model_weight_path="model.pth"

#     test_dataset = MyCityscapesDataset('data/cityscapes', split='val', mode='fine',
#                         target_type='semantic',transform=transform, target_transform=transform_mask)
#     test_loader=DataLoader(test_dataset, batch_size=12, shuffle=False)

#     sample = 2
#     image, mask = next(iter(test_loader))
#     print(image.shape, mask.shape)

#     segmentor = Segmentor(model_weight_path, device='cpu', backbone_name='resnet34')
#     output = segmentor.predict(image[sample])

#     segment = encode_segmap(mask[sample].clone())
#     segment = segment.unsqueeze(0)
#     print("encoded mask:", segment.shape)
    
#     import torch.nn.functional as F
#     loss = F.cross_entropy(output, segment.long())
#     print("loss:", loss.item())

#     criterion= smp.losses.DiceLoss(mode='multiclass')
#     loss=criterion(output,segment.long())
#     print("dice loss:", loss.item())


#     decoded_output=segmentor.decode_pred_mask(output)
#     print("decoded mask:", decoded_output.shape)
    

#     # VISUALIZE MASK
#     decoded_mask=segmentor.preprocess_mask(mask[sample])

#     fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
#     axes[0].imshow(decoded_mask)
#     axes[0].set_title('Ground Truth Mask')
#     axes[0].axis('off')
#     axes[1].imshow(decoded_output)
#     axes[1].set_title('Predicted Mask')
#     axes[1].axis('off')
#     plt.show()

