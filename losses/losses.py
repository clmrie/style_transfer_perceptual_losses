import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

class VGG16LossNetwork(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16LossNetwork, self).__init__()

        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        relu1_2 = self.slice1(x)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)

        return {
            'relu1_2': relu1_2,
            'relu2_2': relu2_2,
            'relu3_3': relu3_3,
            'relu4_3': relu4_3
        }

def compute_content_loss(output_feature, content_feature):
    return F.mse_loss(output_feature, content_feature)

def gram_matrix(feature):
    (b, c, h, w) = feature.size()
    features = feature.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

def compute_style_loss(output_features, style_features, layers=None):
    # Use all keys if layers not specified
    if layers is None:
        layers = output_features.keys()
    
    total_loss = 0.0
    for layer in layers:
        out_feats = output_features[layer]
        style_feats = style_features[layer]
        
        # Compute Gram matrices for both output and style features
        gram_out = gram_matrix(out_feats)
        gram_style = gram_matrix(style_feats)
        
        # Duplicate style Gram matrix if batch sizes do not match
        if gram_out.size(0) != gram_style.size(0):
            gram_style = gram_style.expand(gram_out.size())
        
        layer_loss = F.mse_loss(gram_out, gram_style)
        total_loss += layer_loss
    
    return total_loss


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) + \
           torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    return loss

