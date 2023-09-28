import torch
import torch.nn as nn

from .parts import ConvLSTM
from .unet import Unet


class UnetLstm(Unet):
    def __init__(self, **kwargs):

        super(UnetLstm, self).__init__()

        if not hasattr(self, "n_lstm_layers"):
            self.n_lstm_layers = 1

        self.convLSTM = ConvLSTM(
            in_channels=self.encoder.in_out_sizes[-1]["in_channels"], 
            num_features=self.encoder.in_out_sizes[-1]["in_channels"], 
            kernel_size=3, 
            padding=1, 
            stride=1,
            n_layers=self.n_lstm_layers)
        

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

        if  config["n_central_slices"] == -1: # volumes have different n of slices.
            # Separate volumes again before passing each of them individually to LSTM
            bottle_necks = torch.split(bottle_necks, n_slices, dim=0)

            spatial_encodings = []
            for bottle_neck in bottle_necks:
                spatial_encodings.append(self.convLSTM(bottle_neck.unsqueeze(0)).squeeze(0))
            
            # Re-concatenate 
            spatial_encodings = torch.cat(spatial_encodings, dim=0)
        else:
            shape = (bottle_necks.shape[0]//n_slices[0], n_slices[0],) + tuple(bottle_necks.size()[1:])
            spatial_encodings = self.convLSTM(bottle_necks.view(shape)).view(bottle_necks.shape)

        upsampled = self.decoder(spatial_encodings, out_for_skip_con, config["device"])
        output_masks = self.clf(upsampled)

        return output_masks
    
class ModelLoader(UnetLstm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)




