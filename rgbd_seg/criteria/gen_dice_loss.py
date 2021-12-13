import torch
import torch.nn as nn


class GDL_CrossEntropy(nn.Module):
    """
    Gerneralized Diss Loss + cross entropy loss, for semantic segmentation.
    Arguments:
        num_classes: the number of target classes.
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, num_classes=6, ignore_label=-1, weight=None):
        super(GDL_CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )
        self.smooth_nr = 1e-5
        self.smooth_dr = 1e-5

    def w_func(self, grnd):
        return torch.reciprocal(grnd * grnd)

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)

        l_size = logits.size()
        labels_par_classes = torch.zeros(l_size, dtype=int)
        batch_size = logits.size(0)

        for image in range(batch_size):
            for classe in range(self.num_classes):
                labels_par_classes[image][classe][torch.where(labels[image] == classe)] = 1

        labels = labels_par_classes.cuda()

        reduce_axis = (2, 3)

        m = torch.nn.Softmax(dim=1)
        logits = m(logits)

        intersection = torch.sum(labels * logits, reduce_axis)

        ground_o = torch.sum(labels, reduce_axis)
        pred_o = torch.sum(logits, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f: torch.Tensor = 1.0 - (2.0 * (intersection * w).sum(0) + self.smooth_nr) / (
                (denominator * w).sum(0) + self.smooth_dr)
        f.requires_grad_(True)

        return torch.mean(f) + pixel_losses.mean()