import torch
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt


def predictTTA(model, data, config:dict=None):
    rot90 = T.RandomRotation((90,90))
    rot180 = T.RandomRotation((180,180))
    rot270 = T.RandomRotation((270,270))
    flip = T.RandomHorizontalFlip(p=1)
    all_preds = []
    if config is not None:
        # original
        all_preds.append(model(data, config))

        # original & horizontal flip
        pred = model([flip(volume) for volume in data], config)
        all_preds.append(flip(pred.detach().cpu()).to(config["device"]))

        # 90° rotation
        pred = model([rot90(volume.detach().cpu()).to(config["device"]) for volume in data], config)
        all_preds.append(rot270(pred.detach().cpu()).to(config["device"]))

        # 90° rotation & horizontal flip
        pred = model([flip(rot90(volume.detach().cpu())).to(config["device"]) for volume in data], config)
        all_preds.append(rot270(flip(pred.detach().cpu())).to(config["device"]))

        # 180° rotation
        pred = model([rot180(volume.detach().cpu()).to(config["device"]) for volume in data], config)
        all_preds.append(rot180(pred.detach().cpu()).to(config["device"]))

        # 180° rotation & horizontal flip
        pred = model([flip(rot180(volume.detach().cpu())).to(config["device"]) for volume in data], config)
        all_preds.append(flip(rot180(pred.detach().cpu())).to(config["device"]))

        # 270° rotation
        pred = model([rot270(volume.detach().cpu()).to(config["device"]) for volume in data], config)
        all_preds.append(rot90(pred.detach().cpu()).to(config["device"]))

        # 270° rotation & horizontal flip
        pred = model([flip(rot180(volume.detach().cpu())).to(config["device"]) for volume in data], config)
        all_preds.append(rot90(flip(pred.detach().cpu())).to(config["device"]))
    else:
        # original
        all_preds.append(model(data))

        # original & horizontal flip
        pred = model(flip(data))
        all_preds.append(flip(pred))

        # 90° rotation
        pred = model(rot90(data))
        all_preds.append(rot270(pred))

        # 90° rotation & horizontal flip
        pred = model(flip(rot90(data)))
        all_preds.append(rot270(flip(pred)))

        # 180° rotation
        pred = model(rot180(data))
        all_preds.append(rot180(pred))

        # 180° rotation & horizontal flip
        pred = model(flip(rot180(data)))
        all_preds.append(flip(rot180(pred)))

        # 270° rotation
        pred = model(rot270(data))
        all_preds.append(rot90(pred))

        # 270° rotation & horizontal flip
        pred = model(flip(rot270(data)))
        all_preds.append(rot90(flip(pred)))

    all_preds = torch.stack(all_preds)
    return torch.mean(all_preds, 0)

    
"""
def predictTTA(model, data):

    all_res = []
    rotations = [0,1,2,3]

    model.eval()
    for i in range(data.shape[0]):

        res = []
        for rot in rotations:

            #roitation:
            pred = torch.rot90(model(torch.rot90(data[i:i+1], rot, dims=(2,3))), (4-rot), dims=(2,3)).to('cpu').detach().numpy()
            res.append( pred[0] )

            #roation with reflection
            pred = torch.flip(torch.rot90(model(torch.rot90(torch.flip(data[i:i+1], dims=[2,]), rot, dims=(2,3))), (4-rot), dims=(2,3)), dims=[2,]).to('cpu').detach().numpy()
            res.append( pred[0] )

        all_res.append( np.array(res) )

    all_res = np.array( all_res )

    return all_res
"""