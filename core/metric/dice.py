import torch
from abc import ABC, abstractmethod



class LossBase(ABC):
    def __init__(self, output_transform=lambda x: x):
        super(LossBase, self).__init__()
        self.output_transform = output_transform

    @abstractmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        params = self.output_transform(args)
        return self.forward(*params)



class Loss(LossBase):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self.loss_fn = loss_fn

    def forward(self, *args):
        return self.loss_fn(*args)
    


class Dice(LossBase):
    def __init__(self, smooth=1e-6, output_transform=lambda x: x):
        super(Dice, self).__init__(output_transform)
        self.smooth = smooth

    def forward(self, pred, target):

        target = target.to(pred.dtype)
        sample_indices = target.reshape(target.shape[0], -1).sum(dim=-1).nonzero().flatten()
        pred, target = pred[sample_indices], target[sample_indices]

        inter = pred * target
        inter = inter.reshape(inter.shape[0], -1).sum(dim=1)
        pred = pred.reshape(pred.shape[0], -1).sum(dim=1)
        target = target.reshape(target.shape[0], -1).sum(dim=1)

        dice = ((2 * inter + self.smooth) / (pred + target + self.smooth))
        return 1 - dice
        


class MultiDice(LossBase):
    def __init__(self, class_weight=None, class_weight_alpha=None, smooth=1e-6, is_multilabel=False, output_transform=lambda x: x):
        super(MultiDice, self).__init__(output_transform)

        if (class_weight is None) == (class_weight_alpha is None):
            raise ValueError('If `class_weight` is provided, then `class_weight_alpha` should be not provided and vice versa.')

        self.class_weight = class_weight
        self.class_weight_alpha = class_weight_alpha

        if self.class_weight_alpha is not None and (self.class_weight_alpha < 0 or self.class_weight_alpha > 1):
            raise ValueError('`class_weight_alpha` should be in [0, 1], got {} instead.'.format(self.class_weight_alpha))

        self.smooth = smooth
        self.is_multilabel = is_multilabel

    def forward(self, pred, target):
        target = target.to(pred.device)
        n_samples = target.shape[0]

        if not self.is_multilabel:
            target = torch.zeros_like(pred).scatter(dim=1, index=target.unsqueeze(dim=1), value=1)

        # Count number of pixels of each class
        classes_pixels = target.reshape(*target.shape[:2], -1).sum(dim=-1, dtype=torch.float)
        classes_pixels[classes_pixels == 0] = float('inf')

        # Calculate class weight
        if self.class_weight is not None:
            class_weight = self.class_weight
            class_weight = torch.stack([class_weight] * n_samples, dim=0)
        else:
            # Calculate uniform class weight
            uniform_class_weight = torch.ones_like(classes_pixels, dtype=torch.float) / classes_pixels.size(dim=1)

            # Calculate dynamic class weight
            dynamic_class_weight = 1 / classes_pixels
            dynamic_class_weight /= dynamic_class_weight.sum(dim=1, keepdim=True)

            # Calculate class weight
            class_weight = self.class_weight_alpha * (dynamic_class_weight - uniform_class_weight) + uniform_class_weight
        class_weight = class_weight.to(pred.dtype).to(pred.device) * classes_pixels.size(dim=1)
        inter = pred * target

        inter = inter.reshape(*inter.shape[:2], -1).sum(dim=2)
        pred = pred.reshape(*pred.shape[:2], -1).sum(dim=2)
        target = target.reshape(*target.shape[:2], -1).sum(dim=2)

        dice = (2 * inter + self.smooth) / (pred + target + self.smooth)
        # indices = target.reshape(*target.shape[:2], -1).sum(dim=2).flatten().nonzero(as_tuple=True)
        # dice, class_weight = dice.flatten()[indices], class_weight.flatten()[indices]
        dice, class_weight = dice.flatten(), class_weight.flatten()

        dice = dice * class_weight
        dice = (dice.sum() + self.smooth) / (class_weight.sum() + self.smooth)

        return 1 - dice