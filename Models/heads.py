import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .parts import ConvBlock


class SegmentationHead(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 n_classes:int, 
                 return_logits:bool,
                 kernel_size:int=3,
                 ):
        super(SegmentationHead, self).__init__()
        
        modules = []
        modules.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                shrink=1))

        modules.extend([
            nn.Conv2d(in_channels // 2, n_classes, kernel_size=kernel_size, padding="same"),
            nn.Softmax(dim=1) if not return_logits else nn.Identity()])

        self.module = nn.Sequential(*modules)
        
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return self.module(input)
