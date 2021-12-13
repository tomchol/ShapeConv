import torch.nn as nn

from rgbd_seg.utils import Registry
from rgbd_seg.criteria.gen_dice_loss import GDL_CrossEntropy

CRITERIA = Registry('criterion')

CrossEntropyLoss = nn.CrossEntropyLoss
CRITERIA.register_module(CrossEntropyLoss)
CRITERIA.register_module(GDL_CrossEntropy)


