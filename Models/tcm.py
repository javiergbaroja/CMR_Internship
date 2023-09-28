import torch
from torch import nn

#  TCM file
class TemporalContextBlock(nn.Module):

    def __init__(self,
                inplanes,
                repeat_mode=True,
                is_position_encoding= True,
                window_size=8,
                detach=False,
                local_mean = True,
                reduce=True):
        super(TemporalContextBlock, self).__init__()
        if repeat_mode:
            is_position_encoding = False
        if is_position_encoding:
            assert window_size in [3, 5], 'only support window_size 3 or 5 if position encoding'
            self.mode = 'mode1'
            self.window_size = window_size
        else:
            if repeat_mode:
                self.mode = 'mode2'
            else:
                self.mode = 'mode3'
        if self.mode == 'mode2' and reduce:
            self.reduce = True
        else:
            self.reduce = False
        self.inplanes = inplanes
        # self.snip_size = snip_size
        self.detach = detach
        self.local_mean = local_mean
        #learable layer init
        if self.mode == 'mode1':
            if self.window_size == 5:
                self.tconv_1_1 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_2 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_3 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_4 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_5 = nn.Conv2d(inplanes, 1, kernel_size=1)

                self.tconv_2_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_2 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_4 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_5 = nn.Conv2d(inplanes, inplanes, kernel_size=1)

                self.tconv_3_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_2 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_4 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_5 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
            elif self.window_size == 3:
                self.tconv_1_1 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_2 = nn.Conv2d(inplanes, 1, kernel_size=1)
                self.tconv_1_3 = nn.Conv2d(inplanes, 1, kernel_size=1)

                self.tconv_2_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_2 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_2_3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)

                self.tconv_3_1 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_2 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
                self.tconv_3_3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        elif self.mode == 'mode2':
            self.tconv_1_c = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.tconv_2_c = nn.Conv2d(inplanes, inplanes, kernel_size=1)
            self.tconv_3_c = nn.Conv2d(inplanes, inplanes, kernel_size=1)

            self.tconv_1_o = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.tconv_2_o = nn.Conv2d(inplanes, inplanes, kernel_size=1)
            self.tconv_3_o = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        elif self.mode == 'mode3':
            self.tconv_1 = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.tconv_2 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
            self.tconv_3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        else:
            raise EnvironmentError

        self.instancenorm = nn.InstanceNorm3d(inplanes)
        self.global_conv = torch.nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.global_relu = torch.nn.ReLU(inplace=True)
        # self.global_layernorm=torch.nn.LayerNorm([])
        if not self.reduce:
            self.global_groupnorm = nn.GroupNorm(1,inplanes)
        else:
            self.global_groupnorm = nn.GroupNorm(1,inplanes)

        self.temporal_softmax = nn.Softmax(1)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     if self.mode == 'mode1':
    #         for i in range(self.window_size):
    #             normal_init(getattr(self, 'tconv_1_{}'.format(i+1)))
    #             normal_init(getattr(self, 'tconv_2_{}'.format(i+1)))
    #             normal_init(getattr(self, 'tconv_3_{}'.format(i+1)))
    #     elif self.mode == 'mode2':
    #         normal_init(getattr(self, 'tconv_1_o'))
    #         normal_init(getattr(self, 'tconv_2_o'))
    #         normal_init(getattr(self, 'tconv_3_o'))

    #         normal_init(getattr(self, 'tconv_1_c'))
    #         normal_init(getattr(self, 'tconv_2_c'))
    #         normal_init(getattr(self, 'tconv_3_c'))
    #     elif self.mode == 'mode3':
    #         normal_init(getattr(self, 'tconv_1'))
    #         normal_init(getattr(self, 'tconv_2'))
    #         normal_init(getattr(self, 'tconv_3'))
    #     last_zero_init(self.global_conv)
    #process for mode1
    def process_each(self, feature_maps, position): # postion 0-4 which postion is main
        B, S, C, H, W = feature_maps.size()  # batch size
        # print(feature_maps.size())

        n = (self.window_size-1)//2
        assert S == self.window_size-abs(n-position)

        if self.window_size == 5:
            if position == 1:
                addition = 1
            elif position == 0:
                addition = 2
            else:
                addition = 0
        if self.window_size == 3:
            if position == 0:
                addition = 1
            else:
                addition = 0

        step2s = []
        for i in range(S):
            step2 = getattr(self, 'tconv_2_{}'.format( i+1+addition)) (feature_maps[:,i,...]) # [B,C,H,W]
            step2s.append(step2.view(B,C,-1))
        step2s = torch.stack(step2s,1) #[B,S,C,H*W]
        assert step2s.size() == (B,S,C,H*W),'step2s size not match'
        step1s = []
        for i in range(S):
            step1 = getattr(self,'tconv_1_{}'.format(i+1+addition ) )(feature_maps[:,i,...]) # [B,1,H,W]
            step1s.append(step1.view(B,1,-1))
        step1s = torch.stack(step1s,1)
        step1s = self.temporal_softmax(step1s) # [B,S,1,H*W]
        assert step1s.size() == (B,S,1,H*W),'step1s size not match'
        if self.local_mean:
            step1s = (step1s*step1s.mean(-1,keepdim=True)).permute(0,1,3,2) # [B,S,H*W,1]
        else:
            step1s = step1s.permute(0,1,3,2)

        assert step1s.size() == (B,S,H*W,1),'step1s before matmul size not match'

        step2s = torch.matmul(step2s,step1s) # [B,S,C,1]
        assert step2s.size() == (B,S,C,1),'step2s after matmul size not match'

        feature_maps = feature_maps + step2s.unsqueeze(-1) #[B,S,C,H,W]
        ###########################
        # add layernorm here
        feature_maps = feature_maps.permute(0,2,1,3,4) #[B,C,S,H,W]
        feature_maps = self.instancenorm(feature_maps)

        assert feature_maps.size() == (B,C,S,H,W),'feature_maps after matmul size not match'

        output = getattr(self,'tconv_3_{}'.format(1+addition ) ) (feature_maps[:,:,0,...]) # [B,C,H,W]
        for i in range(1,S):
            output += getattr(self,'tconv_3_{}'.format(i+1+addition ) ) (feature_maps[:,:,i,...]) # [B,C,H,W]


        return output
    def seprate_conv_stack(self,feature_maps,main,conv):
        B,S,C,H,W = feature_maps.size()
        step2so=getattr(self,'tconv_{}_o'.format(conv))(feature_maps.view(-1,C,H,W))
        step2sc=getattr(self,'tconv_{}_c'.format(conv))(main)
        C = 1 if conv == 1 else C
        step2sc = step2sc.unsqueeze(1)
        step2so=step2so.view(B,S,C,H,W)
        assert step2so.size()==(B,S,C,H,W),'step2so size not match'
        assert step2sc.size()==(B,1,C,H,W),'step2so size not match'

        return torch.cat([step2so,step2sc],1)

    def process_each_all(self, feature_maps, position): # only in mode2 and mode3

        B, S, C, H, W = feature_maps.size()  # batch size
        if self.mode =='mode2':
            main=feature_maps[:,position,...] #[B,C,H,W]
            feature_maps=torch.cat([feature_maps[:,:position,...],feature_maps[:,position+1:,...]],1) #[B, S-1,C,H,W]
            step2s=self.seprate_conv_stack(feature_maps,main,2)
            assert step2s.size() == (B,S,C,H,W)

            step1s=self.seprate_conv_stack(feature_maps,main,1)
            assert step1s.size() == (B,S,1,H,W)
        elif self.mode == 'mode3':
            step1s=self.tconv_1(feature_maps.view(B*S,C,H,W)).view(B,S,1,H,W)
            step2s=self.tconv_2(feature_maps.view(B*S,C,H,W)).view(B,S,C,H,W)
            assert step2s.size() == (B,S,C,H,W)
            assert step1s.size() == (B,S,1,H,W)

        step1s = step1s.view(B,S,1,H*W)
        step2s = step2s.view(B,S,C,H*W)
        step1s=self.temporal_softmax(step1s) # [B,S,1,H*W]
        assert step1s.size()==(B,S,1,H*W),'step1s size not match'

        if self.local_mean:
            step1s=(step1s*step1s.mean(-1,keepdim=True)).permute(0,1,3,2) # [B,S,H*W,1]
        else:
            step1s=step1s.permute(0,1,3,2)

        assert step1s.size()==(B,S,H*W,1),'step1s before matmul size not match'
        device = step1s.device
        step2s=torch.matmul(step2s.cpu(), step1s.cpu()).to(device) # [B,S,C,1]
        assert step2s.size()==(B,S,C,1),'step2s after matmul size not match'
        del step1s
        if self.mode=='mode2':
            feature_maps = torch.cat([feature_maps,main.unsqueeze(1)],1) + step2s.unsqueeze(-1) #[B,S,C,H,W]
        elif self.mode == 'mode3':
            feature_maps = feature_maps+step2s.unsqueeze(-1)

        ###########################
        # add layernorm here
        feature_maps=feature_maps.permute(0,2,1,3,4) #[B,C,S,H,W]
        feature_maps=self.instancenorm(feature_maps)
        assert feature_maps.size()==(B,C,S,H,W),'feature_maps after matmul size not match'
        if self.mode=='mode2':
            output=self.tconv_3_o(feature_maps[:,:,:-1,...].permute(0,2,1,3,4).contiguous().view(-1,C,H,W))
            output=output.view(B,S-1,C,H,W).sum(1)
            output+=self.tconv_3_c(feature_maps[:,:,-1,...])
        elif self.mode=='mode3':
            feature_maps=feature_maps.permute(0,2,1,3,4).contiguous().view(-1,C,H,W)
            feature_maps=self.tconv_3(feature_maps)
            output=feature_maps.view(B,S,C,H,W).sum(1)
        return output
    def forward(self,x,snip:int=8):
        B, S, C, H, W = x.shape
        
        if not self.reduce:
            indentity = x
        else:
            indentity = x
            indentity = indentity[:,(S-1)//2,...]

        assert C == self.inplanes, 'channel unmatch'
        
        outputs=[]
        if self.mode=='mode1':
            if self.window_size==5:
                for i in range(S):
                    if i==0:
                        feature_maps=x[:,0:3,...]
                        position=0

                    elif i==1:
                        feature_maps=x[:,0:4,...]
                        position=1

                    elif i==S-2:
                        feature_maps=x[:,i-2:,...]
                        position=3

                    elif i==S-1:
                        feature_maps=x[:,i-2:,...]
                        position=4

                    else:
                        feature_maps=x[:,i-2:i+3,...]
                        position=2
                    if self.detach:
                        feature_maps=feature_maps.detach()
                    output=self.process_each(feature_maps,position)
                    outputs.append(output)

            elif self.window_size==3:
                for i in range(S):
                    if i==0:
                        feature_maps=x[:,0:2,...]
                        position=0
                    elif i==S-1:
                        feature_maps=x[:,i-1:,...]
                        position=2
                    else:
                        feature_maps=x[:,i-1:i+2,...]
                        position=1
                    if self.detach:
                        feature_maps=feature_maps.detach()
                    output=self.process_each(feature_maps,position)
                    outputs.append(output)
        elif self.mode == 'mode2':
            if not self.reduce:
                for i in range(S):
                    feature_maps=x
                    output=self.process_each_all(feature_maps,i)
                    outputs.append(output)
            else:
                # assert (S-1)%2==0,'reduce mode must have 2n+1 input frames'
                feature_maps=x
                output=self.process_each_all(feature_maps,int((S-1)//2))
                output=self.global_groupnorm(output)
                output=self.global_relu(output) #[B,C,H,W]
                output=self.global_conv(output)
                output +=indentity
                return output
        elif self.mode =='mode3':
            for i in range(S):
                feature_maps=x
                output=self.process_each_all(feature_maps,i)
                outputs.append(output)


        outputs=torch.stack(outputs,1)
        assert outputs.size()==(B, S, C, H, W)
        outputs=self.global_groupnorm(outputs.view(B*S, C, H, W))
        outputs=self.global_relu(outputs)
        outputs=self.global_conv(outputs)
        outputs = indentity + outputs.view(B, S, C, H, W)
        return outputs
