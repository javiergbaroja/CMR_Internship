import torch
import torch.nn as nn

from .parts import *
from .heads import SegmentationHead


class Unet(nn.Module):
    def __init__(self, **kwargs):

        super(Unet, self).__init__()

        for k,v in kwargs.items():
            setattr(self, k, v)

        self._get_encoder(**kwargs)

        self.decoder = Decoder(
            self.decoder_channels[0], 
            self.decoder_channels[1],
            self.interp,
            self.how_to_merge_skip_con)
        
        self.clf = SegmentationHead(
            in_channels=self.decoder.in_out_sizes[-1]["out_channels"], 
            n_classes=self.n_classes,
            return_logits=self.return_logits)
        

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

        # if  config["n_central_slices"] == -1: # volumes have different n of slices.
        #     # Separate volumes again before passing each of them individually to LSTM
        #     bottle_necks = torch.split(bottle_necks, n_slices, dim=0)

        #     spatial_encodings = []
        #     for bottle_neck in bottle_necks:
        #         spatial_encodings.append(self.convLSTM(bottle_neck.unsqueeze(0)).squeeze(0))
            
        #     # Re-concatenate 
        #     spatial_encodings = torch.cat(spatial_encodings, dim=0)
        # else:
        #     shape = (bottle_necks.shape[0]//n_slices[0], n_slices[0],) + tuple(bottle_necks.size()[1:])
        #     spatial_encodings = self.convLSTM(bottle_necks.view(shape)).view(bottle_necks.shape)

        upsampled = self.decoder(bottle_necks, out_for_skip_con, config["device"])
        output_masks = self.clf(upsampled)

        return output_masks
    
    def _get_encoder(self, **kwargs):
        if self.encoder == "resnet34":
            self.interp = "nearest"
            self.how_to_merge_skip_con = "concat"
            self.encoder = ResNetEncoder(self.encoder,**kwargs)
        elif self.encoder == "small":
            self.interp = "bilinear"
            self.how_to_merge_skip_con = "add"
            self.encoder =  SmallEncoder(**kwargs)
        else:
            raise ValueError(f"The encoder model architecture must be chosen from ['small', 'resnet34']")
        
        self.decoder_channels = self.encoder.get_skip_connection_channel_sizes()
    
class ModelLoader(Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)




