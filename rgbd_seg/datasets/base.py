from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    """ BaseDataset
    """
    CLASSES = None

    PALETTE = None

    def __init__(self, transform=None, transform_rgb=None):
        self.transform = transform
        self.transform_rgb = transform_rgb

    def process(self, image, masks):
        if self.transform_rgb:
            img_rgb = image[:, :, :3].astype(np.uint8)
            augmented_rgb = self.transform_rgb(image=img_rgb)['image']
            image[:, :, :3] = augmented_rgb.astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            return augmented['image'], augmented['masks']
        else:
            return image, masks
