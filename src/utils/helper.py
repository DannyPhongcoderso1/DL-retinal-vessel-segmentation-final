import numpy as np

def normalize_mask(mask, **kwargs):
    return mask.astype(np.float32) / 255.0

def tensor_to_image(tensor):
    img = tensor.numpy().transpose(1, 2, 0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    
    img = np.clip(img, 0, 1)
    return img