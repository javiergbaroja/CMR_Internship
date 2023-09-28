import torch
import torch.nn.functional as F
import torch.nn as nn

from .parts import ConvBiLSTM, Decoder
from .unet import Unet


class UnetMultiBiLstm(Unet):
    """This segmentation network has a Conv-BiLSTM block in each skip connection. 
    The LSTM receives the concatenation of the upsampled feature maps from the decoder and the feature maps from the encoder's skip connection.
    """

    def __init__(self, **kwargs):

        super(UnetMultiBiLstm, self).__init__(**kwargs)

        self.convBiLSTM_bottleneck = ConvBiLSTM(
            in_channels=self.encoder.in_out_sizes[-1]["in_channels"], 
            num_features=self.encoder.in_out_sizes[-1]["in_channels"], 
            kernel_size=3, 
            padding=1, 
            stride=1,
            merge_mode=self.merge_bilstm,
            n_layers=self.n_lstm_layers)
        
        self.smaller_decoder_channels = [self.decoder_channels[0][i] - self.decoder_channels[1][i] for i in range(len(self.decoder_channels[0]))]
        
        self.convBiLSTMs_upsampling = nn.ModuleList([ConvBiLSTM(in_channels=self.decoder_channels[0][i], 
                                                                num_features=self.smaller_decoder_channels[i], 
                                                                kernel_size=3, 
                                                                padding=1, 
                                                                stride=1,
                                                                merge_mode=self.merge_bilstm,
                                                                n_layers=1) for i in range(len(self.decoder_channels[0]))])
        
        self.decoder = Decoder(
            self.smaller_decoder_channels, 
            self.decoder_channels[1],
            self.interp,
            self.how_to_merge_skip_con)
        

    def upsample(self, input_small:torch.Tensor, skip_con:list, device:torch.device, n_slices:int) -> torch.Tensor:

        n_layers = len(skip_con)
        if len(self.decoder.convLayers) == len(skip_con):
            adjust_pos = 1
        elif len(self.decoder.convLayers) == len(skip_con) - 1:
            adjust_pos = 2

        for i in range(n_layers):
            input_small = F.interpolate(input_small.to("cpu"), scale_factor=2, mode=self.decoder.interp_mode).to(device)
            if i < len(self.decoder.convLayers):
                if self.decoder.join_mode == "concat":
                    input_small = torch.cat((input_small, skip_con[n_layers-i-adjust_pos]), dim=1)
                    # TODO: here goes the forward for the conv-LSTM
                    shape_stacked = (input_small.shape[0], self.smaller_decoder_channels[i]) + tuple(input_small.size()[2:])
                    shape_separated = (input_small.shape[0]//n_slices, n_slices,) + tuple(input_small.size()[1:])
                    input_small = self.convBiLSTMs_upsampling[i](input_small.view(shape_separated)).view(shape_stacked)
                elif self.decoder.join_mode == "add":
                    input_small += skip_con[n_layers-i-adjust_pos]
                
                input_small = self.decoder.convLayers[i](input_small)
        
        return input_small
        

    def forward(self, inputs:list, config:dict) -> torch.Tensor:
        """Forward pass through UNET-LSTM

        Args:
            inputs (list): The list contains separate patients. Their total number of slices meets the batch_size requirement.
                Each item in the list of shape (Slices, Channels, Height, Width)
            device (torch.device): Interpolation operation is not supported in GPU. 
                This argument is needed to return the outuput of the upsample operation to the appropriate device.

        Returns:
            torch.Tensor: _description_
        """

        n_slices = [volume.shape[0] for volume in inputs]
        
        # Concat all images for more efficient pass through non-sequential part of network
        inputs = torch.cat(inputs, dim=0)

        out_for_skip_con, bottle_necks = self.encoder(inputs)

        if  n_slices.count(n_slices[0]) != len(n_slices): # volumes have different n of slices.
            # Separate volumes again before passing each of them individually to LSTM
            bottle_necks = torch.split(bottle_necks, n_slices, dim=0)

            spatial_encodings = []
            for bottle_neck in bottle_necks:
                spatial_encodings.append(self.convBiLSTM_bottleneck(bottle_neck.unsqueeze(0)).squeeze(0))
            
            # Re-concatenate 
            spatial_encodings = torch.cat(spatial_encodings, dim=0)
        else:
            shape = (bottle_necks.shape[0]//n_slices[0], n_slices[0],) + tuple(bottle_necks.size()[1:])
            spatial_encodings = self.convBiLSTM_bottleneck(bottle_necks.view(shape)).view(bottle_necks.shape)

        upsampled = self.upsample(spatial_encodings, out_for_skip_con, config["device"], n_slices[0])
        output_masks = self.clf(upsampled)

        return output_masks
    
    
class ModelLoader(UnetMultiBiLstm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)




