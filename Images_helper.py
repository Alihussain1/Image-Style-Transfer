from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def load_image(img_path,max_size,mean,std):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    #normalize(tensor, mean, std)
    #mean (sequence) – Sequence of means for each channel.
    #std (sequence) – Sequence of standard deviations for each channely

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = transform(image)[:3,:,:].unsqueeze(0)
    
    return image
def Tensor_to_Image(tensor,mean,std):
        image = tensor.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * std + mean
        image = image.clip(0, 1)
        return image