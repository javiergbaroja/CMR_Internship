import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

# Resnet Adapted from https://github.com/qubvel/segmentation_models.pytorch

class ConvBlock(nn.Module):
    """Block module describing downsampling architecture (2D CNN)
    """

    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 conv_type:str="same",
                 n_layers:int=2,
                 kernel_size:int=3,
                 stride:int=1,
                 shrink:bool=True):
        
        super(ConvBlock, self).__init__()
        if conv_type not in ["valid", "same"]:
            raise ValueError(f"conv_type argument passed as {conv_type}. Should be one of 'valid' or 'same'.")
        
        self.module = self._build_module(in_channels, out_channels, kernel_size, stride, n_layers, conv_type, shrink)


    def _build_module(self, input_channels, n_channels, kernel, stride, n_layers, conv_type, shrink):
        modules = []
        if conv_type == "valid":
            conv_type = 0

        for i in range(n_layers):
            in_channels = input_channels if shrink and i==0 else n_channels
            modules.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=kernel, stride=stride, padding=conv_type),
                nn.BatchNorm2d(num_features=n_channels),
                nn.ReLU()
            ])

        return nn.Sequential(*modules)

    def forward(self, input:torch.Tensor) -> torch.Tensor:

        return self.module(input)
    
class SmallEncoder(nn.Module):
    def __init__(self, 
                 n_layers:int, 
                 in_channels:int, 
                 out_channels:int, 
                 maxpool_kernel:int=2, 
                 conv_type:str="same", 
                 **kwargs):
        
        super(SmallEncoder, self).__init__()

        self._check_conv_type(conv_type)

        self.out_channels = out_channels
        self.n_layers = n_layers

        intermediate_channels=int(0.75*out_channels)

        self.convblock = [ConvBlock(in_channels, intermediate_channels, conv_type, n_layers=1, shrink=i==0) for i in range(1)][0]
        self.downblock = nn.ModuleList([ConvBlock(intermediate_channels, out_channels, conv_type, shrink=i==0) for i in range(n_layers)])
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel)

        self._save_in_out_sizes()
    
    def forward(self, input:torch.Tensor) -> list:
        out_for_skip_con = []
        input = self.convblock(input)
        for i in range(len(self.downblock)):
            input = self.downblock[i](input)
            out_for_skip_con.append(input)
            input = self.maxpool(input)
        
        return out_for_skip_con, input
        
    def _check_conv_type(self, conv_type:str):
        if conv_type not in ["same", "valid"]:
            raise ValueError(f"Type of convolution can only be ['small', 'valid']")
        self.conv_type = conv_type

    def get_skip_connection_channel_sizes(self):

        in_c = out_c = [self.out_channels for __ in range(self.n_layers)]
        return in_c, out_c
    
    def _save_in_out_sizes(self):
        sizes = []
        for name, module in nn.Sequential(self.convblock, self.downblock).named_modules():
            if isinstance(module, nn.Conv2d):
                sizes.append({
                    "name":name, 
                    "in_channels":module.in_channels,
                    "out_channels":module.out_channels})
        self.in_out_sizes = [sizes[0], sizes[-1]]


class ResNetEncoder(nn.Module):
    def __init__(self, name:str, in_channels:int, pretrained:str=None, **kwargs) -> None:
        super().__init__()

        self.resnet = getattr(models, name)()
        self.in_channels = in_channels

        if pretrained == "imagenet":
            path = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
            self.resnet.load_state_dict(torch.hub.load_state_dict_from_url(path))

        # Remove Fully connected sections
        del self.resnet.avgpool
        del self.resnet.fc

        self._adapt_in_channels(pretrained != None)
        self._save_in_out_sizes()
        self._unfreeze()

    def _unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

    def _save_in_out_sizes(self):
        sizes = []
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                sizes.append({
                    "name":name, 
                    "in_channels":module.in_channels,
                    "out_channels":module.out_channels})
        self.in_out_sizes = [sizes[0], sizes[-1]]


    def _adapt_in_channels(self, is_pretrained:bool):
        """Change first convolution layer input channels.
        For:
            in_channels == 1 or in_channels == 2 -> reuse original weights
            in_channels > 3 -> make random kaiming normal initialization
        """

        # Extract first conv layer
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels != self.in_channels:
                old_in_channels = module.in_channels
                break
        
        if old_in_channels == self.in_channels:
            pass
        else:
            module.in_channels = self.in_channels
            if not is_pretrained:
                module.weight = nn.parameter.Parameter(
                    torch.Tensor(module.out_channels, self.in_channels // module.groups, *module.kernel_size))
                module.reset_parameters()
            else:
                weight = module.weight.detach()
                if self.in_channels == 1:
                    new_weight = weight.sum(1, keepdim=True)
                    module.weight = nn.parameter.Parameter(new_weight)
                
                elif self.in_channels < old_in_channels:
                    new_weight = weight[:, :self.in_channels] * (old_in_channels / self.in_channels)

                else:
                    new_weight = torch.Tensor(module.out_channels, (self.in_channels // module.groups) - old_in_channels, *module.kernel_size)
                    module.weight = nn.parameter.Parameter(torch.concatenate((weight, new_weight), dim=1))
            

    def forward(self, input:torch.Tensor) -> tuple:

        out_for_skip_con = []
        for name,child in self.resnet.named_children():
            input = child(input)
            if "relu" in name or "layer" in name:
                out_for_skip_con.append(input)

        return out_for_skip_con, input
    
    def get_skip_connection_channel_sizes(self):

        in_c =  [768, 384, 192, 128]
        out_c = [256, 128,  64,  32]
        return in_c, out_c
    
class Decoder(nn.Module):
    """Block module describing upsampling architecture (2D CNN)
    """

    def __init__(self, 
                 in_channels:list, 
                 out_channels:list,
                 interp_mode:str="nearest",
                 join_mode:str='concat'):
        
        super(Decoder, self).__init__()
        assert len(in_channels) == len(out_channels)
        self.n_layers = len(in_channels)

        self.interp_mode = interp_mode
        self.join_mode = join_mode
        self.convLayers = nn.ModuleList([ConvBlock(in_channels[i], out_channels[i], shrink=True,conv_type="same") for i in range(self.n_layers)])
        self._save_in_out_sizes()

    def forward(self, input_small:torch.Tensor, skip_con:list, device):

        n_layers = len(skip_con)
        if len(self.convLayers) == len(skip_con):
            adjust_pos = 1
        elif len(self.convLayers) == len(skip_con) - 1:
            adjust_pos = 2

        for i in range(n_layers):
            input_small = F.interpolate(input_small.to("cpu"), scale_factor=2, mode=self.interp_mode).to(device)
            if i < len(self.convLayers):
                if self.join_mode == "concat":
                    input_small = torch.cat((input_small, skip_con[n_layers-i-adjust_pos]), dim=1)
                elif self.join_mode == "add":
                    input_small += skip_con[n_layers-i-adjust_pos]
                
                input_small = self.convLayers[i](input_small)
        
        return input_small
    
    def _save_in_out_sizes(self):
        sizes = []
        for name, module in self.convLayers.named_modules():
            if isinstance(module, nn.Conv2d):
                sizes.append({
                    "name":name, 
                    "in_channels":module.in_channels,
                    "out_channels":module.out_channels})
        self.in_out_sizes = [sizes[0], sizes[-1]]


class ConvLSTM(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_features:int, 
                 kernel_size:int=3, 
                 padding:int=1, 
                 stride:int=1,
                 n_layers:int=1) -> None:
        super().__init__()

        self.n_layers = n_layers
        if n_layers == 1:
            self.conv_lstm = ConvLSTMLayer(in_channels, num_features, kernel_size, padding, stride)
        else:
            self.conv_lstm = nn.Sequential(*[ConvLSTMLayer(in_channels, num_features, kernel_size, padding, stride) for __ in range(n_layers)])
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv_lstm(x)
    

class ConvBiLSTM(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_features: int, 
                 kernel_size: int = 3, 
                 padding: int = 1, 
                 stride: int = 1, 
                 merge_mode: str = "concat",
                 n_layers: int = 1):
        super(ConvBiLSTM, self).__init__()

        self.n_layers = n_layers
        if n_layers == 1:
            self.conv_bilstm = ConvBiLSTMLayer(in_channels, num_features, kernel_size, padding, stride, merge_mode)
        else:
            self.conv_bilstm = nn.Sequential(*[ConvBiLSTMLayer(in_channels, num_features, kernel_size, padding, stride, merge_mode) for __ in range(n_layers)])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv_bilstm(x)


class ConvLSTMLayer(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 num_features:int, 
                 kernel_size:int=3, 
                 padding:int=1, 
                 stride:int=1):
        
        super(ConvLSTMLayer, self).__init__()
        self.num_features = num_features
        self._make_layer(in_channels+num_features, num_features*4, kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            gates = self.norm(gates)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)
    

class ConvBiLSTMLayer(nn.Module):
    def __init__(self, in_channels: int, num_features: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, merge_mode:str="concat"):
        super(ConvBiLSTMLayer, self).__init__()

        self.merge_mode = merge_mode

        if self.merge_mode == "concat":
            num_features_forward = num_features//2 
            num_features_backward = num_features - num_features_forward 
        else:
            num_features_forward = num_features
            num_features_backward = num_features

        self.lstm_forward = ConvLSTMLayer(in_channels, num_features_forward, kernel_size, padding, stride)
        self.lstm_backward = ConvLSTMLayer(in_channels, num_features_backward, kernel_size, padding, stride)
    
    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
    
        out_forward = self.lstm_forward(inputs)
        out_backward = self.lstm_backward(torch.flip(inputs, [1]))

        if self.merge_mode == "concat":
            return torch.cat((out_forward, out_backward), dim=2) # (B, S, C, H, W)
        
        elif self.merge_mode == "sum":
            return out_forward + out_backward
        
        elif self.merge_mode == "average":
            return torch.mean(torch.stack([out_forward, out_backward]))
        
        elif self.merge_mode == "multiply":
            return out_forward * out_backward






