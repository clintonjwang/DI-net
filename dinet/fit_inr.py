import os
import sys
import torch
osp = os.path
import numpy as np

from dinet.utils import args as args_module
from dinet.utils import losses
from dinet.data import dataloader
from dinet.inrs import siren, rff
from dinet.data import inet
from dinet import DS_DIR

def fit_siren_to_data(data, total_steps, **kwargs):
    return fit_siren_to_img(data, total_steps, **kwargs)

def fit_siren_to_img(img, total_steps, **kwargs):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    elif len(img.shape) != 4:
        raise ValueError(f'bad shape {img.shape}')
    imgFit = siren.ImageFitting(img)
    H,W = imgFit.H, imgFit.W
    dl = torch.utils.data.DataLoader(imgFit, batch_size=1, pin_memory=True, num_workers=0)
    inr = siren.Siren(out_channels=img.size(1), **kwargs).cuda()
    optim = torch.optim.Adam(lr=1e-4, params=inr.parameters())
    model_input, ground_truth = next(iter(dl))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output = inr(model_input)    
        loss = (losses.mse_loss(model_output, ground_truth)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("Loss %0.4f" % (loss), flush=True)

    return inr, loss



def train_inr(args, **kwargs):
    dl_args = args["data loading"]
    ds_name = dl_args["dataset"]
    start_ix = int(args["start_ix"])
    end_ix = start_ix+dl_args["end_ix"]
    total_steps = args["optimizer"]["max steps"]

    while osp.exists(DS_DIR+f"/{ds_name}/{dl_args['subset']}_{start_ix}.pt"):
        start_ix += 1
    print("Starting", flush=True)

    if args['network']['type'] == 'SIREN':
        fit_fxn = fit_siren_to_data
    elif args['network']['type'] == 'RFF':
        fit_fxn = rff.fit_rff_to_img

    dataset = dataloader.get_img_dataset(args)
    param_dict = {}
    for ix in range(start_ix,end_ix):
        path = DS_DIR+f"/{ds_name}/{dl_args['subset']}_{ix}.pt"
        if osp.exists(path):
            continue

        data = dataset[ix]
        if isinstance(data, tuple):
            img, data = data[0], data[1:]
        else:
            img = data
        inr,loss = fit_fxn(img, total_steps, **kwargs)
        for k,v in inr.state_dict().items():
            param_dict[k] = v.cpu()

        loss_path = DS_DIR+f"/{ds_name}/loss_{dl_args['subset']}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))

        if dl_args["variables"] is None:
            torch.save(param_dict, path)
        else:
            torch.save((param_dict, data[1:]), path)


def main(args):
    args = args_module.parse_args_fitting(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    if args["data loading"]["dataset"] == "inet12":
        train_inet12(args)
    elif args["data loading"]["dataset"] == "cityscapes":
        train_cityscapes(args)
    else:
        train_inr(args)


def train_inet12(args):
    start_ix = int(args["start_ix"])

    if args['network']['type'] == 'SIREN':
        fit_fxn = fit_siren_to_img
        if start_ix < 800:
            subset='train'
            dataset = inet.INet12(subset=subset)
        else:
            start_ix -= 800
            subset='val'
            dataset = inet.INet12(subset='test')
            
    elif args['network']['type'] == 'RFF':
        subset = 'test'
        dataset = inet.INet12(subset=subset)
        fit_fxn = rff.fit_rff_to_img
        
    end_ix = start_ix+100
    total_steps = args["optimizer"]["max steps"]
    print("Starting", flush=True)

    for j in range(start_ix,end_ix):
        if args['network']['type'] == 'RFF':
            cls, ix = j%12, j // 12
        else:
            cls, ix = j%12, j // 12 + 800

        path = DS_DIR+f"/inet12/{cls}/{subset}_{ix}.pt"
        if osp.exists(path):
            continue

        img = dataset[j]
        inr,loss = fit_fxn(img, total_steps)
        torch.save(inr.cpu().state_dict(), path)
        loss_path = DS_DIR+f"/inet12/{cls}/loss_{subset}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))


def train_cityscapes(args):
    dl_args = args["data loading"]
    start_ix = int(args["start_ix"])
    if start_ix < 2975:
        dl_args['subset']='train'
    elif start_ix < 3475:
        dl_args['subset']='val'
        start_ix -= 2975
    else:
        dl_args['subset']='train_extra'
        start_ix -= 500
        
    end_ix = start_ix+100
    total_steps = args["optimizer"]["max steps"]
    if dl_args['subset'] == 'train_extra':
        subset = 'train'
    else:
        subset = dl_args['subset']

    os.makedirs(DS_DIR+f"/cityscapes", exist_ok=True)
    while osp.exists(DS_DIR+f"/cityscapes/{subset}_{start_ix}.pt"):
        start_ix += 1
    print("Starting", flush=True)

    dataset = dataloader.get_img_dataset(args)
    for ix in range(start_ix,end_ix):
        if dl_args['subset']=='train_extra':
            img,seg = dataset[ix-2975]
        else:
            img,seg = dataset[ix]
        inr,loss = fit_siren_to_img(img, total_steps)
        path = DS_DIR+f"/cityscapes/{subset}_{ix}.pt"
        torch.save((inr.state_dict(), seg.squeeze(0)), path)
        loss_path = DS_DIR+f"/cityscapes/loss_{subset}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))


if __name__ == "__main__":
    main()
