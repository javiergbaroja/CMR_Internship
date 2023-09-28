import torch
import torch.nn as nn

from .parts import ConvBiLSTM, Decoder
from .unet import Unet
from .tcm import TemporalContextBlock
from .transformer import Transformer


class UnetBiLstm(Unet):
    def __init__(self, **kwargs):

        super(UnetBiLstm, self).__init__(**kwargs)

        if not hasattr(self, "merge_bilstm"):
            self.merge_bilstm = "concat"
        if not hasattr(self, "n_lstm_layers"):
            self.n_lstm_layers = 1
        self.decoder_channels[0][0]  = self.decoder_channels[0][0] - self.encoder.in_out_sizes[-1]["in_channels"] + self.encoder.in_out_sizes[-1]["in_channels"]//4
        self.decoder = Decoder(
            self.decoder_channels[0], 
            self.decoder_channels[1],
            self.interp,
            self.how_to_merge_skip_con)

        self.swin_tcm = Transformer(in_channels=self.encoder.in_out_sizes[-1]["in_channels"], 
                                    img_size=4, 
                                    patch_size=1, 
                                    hidden_size=self.encoder.in_out_sizes[-1]["in_channels"]//4, 
                                    num_layers=3, 
                                    mlp_dim=8, 
                                    num_heads=8, 
                                    dropout_rate=0.1, 
                                    attention_dropout_rate=0.1)
        
        # TemporalContextBlock(inplanes=self.encoder.in_out_sizes[-1]["in_channels"], reduce=False)
        

    def forward(self, inputs:list, config:dict) -> torch.Tensor:
        """Forward pass through UNET-LSTM

        Args:
            inputs (list): The list contains separate patients. Their total number of slices meets the batch_size requirement.
                Each item in the list of shape (Slices, Channels, Height, Width)
            device (torch.device): _description_

        Returns:
            torch.Tensor: _description_
        """

        n_slices = [volume.shape[0] for volume in inputs]
        
        # Concat all images for more efficient pass through non-sequential part of network
        inputs = torch.cat(inputs, dim=0)

        out_for_skip_con, bottle_necks = self.encoder(inputs)

        # if  n_slices.count(n_slices[0]) != len(n_slices): # volumes have different n of slices.
            # Separate volumes again before passing each of them individually to LSTM
        bottle_necks = torch.split(bottle_necks, n_slices, dim=0)

        spatial_encodings = []
        for bottle_neck in bottle_necks:
            spatial_encodings.append(self.swin_tcm(bottle_neck.unsqueeze(0)).squeeze(0))
        
        # Re-concatenate 
        spatial_encodings = torch.cat(spatial_encodings, dim=0)
        # else:
        #     shape = (bottle_necks.shape[0]//n_slices[0], n_slices[0],) + tuple(bottle_necks.size()[1:])
        #     spatial_encodings = self.tcm(bottle_necks.view(shape)).view(bottle_necks.shape)

        upsampled = self.decoder(spatial_encodings, out_for_skip_con, config["device"])
        output_masks = self.clf(upsampled)

        return output_masks
    
class ModelLoader(UnetBiLstm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)




