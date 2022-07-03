import torch

from .model import Model
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models.segmentation import fcn


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
}


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True, replace_stride_with_dilation=[False, True, True]):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = fcn.FCNHead(inplanes, num_classes)

    model_map = {
        'fcn': (fcn.FCNHead, fcn.FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


class FCN(Model):
    def __init__(self, backbone, backbone_fixed=False, pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs):
        super(FCN, self).__init__()
        supported_backbone = [
            'resnet50',
            'resnet101',
        ]

        if backbone not in supported_backbone:
            raise ValueError('{} is not supported.'.format(backbone))

        self.model = _load_model('fcn', backbone, pretrained, progress, num_classes, aux_loss, **kwargs)
        self.model.backbone.requires_grad_(not backbone_fixed)

    def forward(self, x):
        output = self.model(x)['out']
        output = torch.nn.functional.softmax(output, dim=1)
        return output
