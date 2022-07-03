import torch

from ..handlers.evaluator import MetricBase


class SegmMetric(MetricBase):
    def __init__(self, metric_name: str = None, output_transform=lambda x: x):
        super(SegmMetric, self).__init__(output_transform)
        self.metric_name = metric_name

    def reset(self):
        self._sum = 0
        self._num_samples = 0

    def update(self, output):
        '''
        Args:
            preds, targets, image_infos = output
            preds: torch.Tensor [B, num_classes, H, W]
            targets: torch.Tensor [B, H, W]
            image_infos: List[Tuple(image_path, (w, h))]
        Outputs:
            ...
        '''
        # assert self.metric_name in ['pixel_accuracy', 'mean_pixel_accuracy',
        #                             'mean_iou', 'frequence_weighted_IU'], f'metric: {self.metric_name} not supported'

        preds, targets, image_infos = output
        _, image_sizes = image_infos
        image_sizes = [(w.item(), h.item()) for w, h in zip(*image_sizes)]

        targets = targets.to(preds.dtype).unsqueeze(dim=1)  # B, 1, H, W
        preds = torch.argmax(preds, dim=1, keepdims=True).to(targets.dtype)  # B, 1, H, W

        value = 0
        for i in range(len(image_sizes)):
            pred, target, image_size = preds[i:i + 1], targets[i:i + 1], image_sizes[i]     # 1, 1, H, W
            pred = torch.nn.functional.interpolate(pred, size=image_size[::-1], mode='nearest')  # 1, 1, H, W
            target = torch.nn.functional.interpolate(target, size=image_size[::-1], mode='nearest')  # 1, 1, H, W
            pred, target = pred.squeeze(dim=0).squeeze(dim=0), target.squeeze(dim=0).squeeze(dim=0)  # pred, target: H, w
            if self.metric_name == 'pixel_accuracy':
                metric = self._pixel_accuracy(pred, target)
            elif self.metric_name == 'mean_pixel_accuracy':
                metric = self._mean_pixel_accuracy(pred, target)
            elif self.metric_name == 'mean_iou':
                metric = self._mean_IU(pred, target)
            elif self.metric_name == 'frequence_weighted_IU':
                metric = self._frequency_weighted_IU(pred, target)
            elif self.metric_name == 'precision':
                metric = self._precision(pred, target)
            elif self.metric_name == 'recall':
                metric = self._recall(pred, target)
            elif self.metric_name == 'f1_score':
                metric = self._f1_score(pred, target)

            value += metric
            self._sum += metric
            self._num_samples += 1
        
        return value / len(image_sizes)        

    def compute(self):
        return self._sum / self._num_samples


    def process(self, pred, target, categories):
        tp = torch.empty(len(categories))
        fp = torch.empty(len(categories))
        fn = torch.empty(len(categories))
        tn = torch.empty(len(categories))
        for i in range(categories.shape[0]):
            tp[i] = ((target == categories[i]) & (pred == categories[i])).sum().item()
            fp[i] = ((target == categories[i]) & (pred != categories[i])).sum().item()
            fn[i] = ((target != categories[i]) & (pred == categories[i])).sum().item()
            tn[i] = ((target != categories[i]) & (pred != categories[i])).sum().item()
        return tp, fp, fn, tn
    
    
    def _precision(self, pred, target):
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))
        tp, fp, _, _ = self.process(pred, target, categories)
        
        precision = tp / (tp + fp)
        return precision.sum().item() / len(precision)
    
    def _recall(self, pred, target):
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))
        tp, _, fn, _ = self.process(pred, target, categories)
        
        recall = tp / (tp + fn)
        return recall.sum().item() / len(recall)
    
    def _f1_score(self, pred, target):
        precision = self._precision(pred, target)
        recall = self._recall(pred, target)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _pixel_accuracy(self, pred, target):
        '''pixel_accuracy = sum_i(n_ii) / sum_i(t_i)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            pixel_accuracy: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))
        sum_n_ii, sum_t_i = 0, 0
        for category in categories:
            sum_n_ii += ((target == category) & (pred == category)).sum().item()    # 
            sum_t_i += (target == category).sum().item()

        pixel_accuracy = sum_n_ii / sum_t_i if sum_t_i != 0 else 0.

        return pixel_accuracy

    def _mean_pixel_accuracy(self, pred, target):
        '''mean_pixel_accuracy = (1/n_cl) * sum_i(n_ii / t_i)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            mean_pixel_accuracy: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        pixel_accs = []
        for category in categories:
            n_ii = ((target == category) & (pred == category)).sum().item()
            t_i = (target == category).sum().item()
            pixel_acc = n_ii / t_i if t_i != 0 else 0.
            pixel_accs.append(pixel_acc)

        mean_pixel_accuracy = sum(pixel_accs) / len(pixel_accs) if len(pixel_accs) != 0 else 0.

        return mean_pixel_accuracy

    def _mean_IU(self, pred, target):
        '''mean_iou = (1 / n_cl) * sum_i(n_ii / (t_i + sum_j(n_ij) - n_ii)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            mean_iou: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        ious = []
        for category in categories:
            inter = ((target == category) & (pred == category)).sum().item()
            union = (target == category).sum().item() + (pred == category).sum().item() - inter
            iou = inter / union
            ious.append(iou)

        mean_iou = sum(ious) / len(ious) if len(ious) != 0 else 0.

        return mean_iou

    def _frequency_weighted_IU(self, pred, target):
        '''frequence_weighted_IU = (1 / sum_k(t_k)) * sum_i(t_i * n_ii / (t_i + sum_j(n_ij) - n_ii)
        Args:
            pred: torch.Tensor [H, W]
            target: torch.Tensor [H, W]
        Outputs:
            frequence_weighted_IU: float
        '''
        pred_categories = torch.unique(pred)
        true_categories = torch.unique(target)
        categories = torch.unique(torch.cat([true_categories, pred_categories], dim=0))

        freq_ious = []
        for category in categories:
            n_ii = ((target == category) & (pred == category)).sum().item()
            t_i = (target == category).sum().item()
            n_ij = (pred == category).sum().item()
            freq_iou = (t_i * n_ii) / (t_i + n_ij - n_ii)
            freq_ious.append(freq_iou)

        sum_k_t_k = target.shape[0] * target.shape[1]

        fw_iou = sum(freq_ious) / sum_k_t_k

        return fw_iou
