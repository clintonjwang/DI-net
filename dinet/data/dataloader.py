import os
import torch
osp=os.path
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
nn=torch.nn
from dinet.inrs import siren
from dinet.utils.util import glob2
from dinet import DS_DIR
from dinet.data import sdf_synthetic, inet, cityscapes
nearest = transforms.InterpolationMode('nearest')


class jpgDS(torchvision.datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.paths = glob2(self.root,"*.jpg")
    def __getitem__(self, ix):
        return self.transform(Image.open(self.paths[ix]))
    def __len__(self):
        return len(self.paths)

def get_img_dataset(args):
    totorch_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
    ])
    dl_args = args["data loading"]
    if dl_args["dataset"] == "imagenet1k":
        dataset = inet.INetDS(DS_DIR+"/imagenet_pytorch", subset=dl_args['subset'])

    elif dl_args["dataset"] == "cityscapes":
        size = dl_args['image shape']
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if dl_args['subset'] == 'train_extra':
            mode = 'coarse'
        else:
            mode = 'fine'
        return torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split=dl_args['subset'], mode=mode, target_type='semantic',
            transform=trans, target_transform=cityscapes.seg_transform)

    else:
        raise NotImplementedError
    return dataset


def get_inr_dataloader(dl_args):
    if dl_args["dataset"] == "sdf_synthetic":
        if dl_args['subset'] == 'balls':
            return sdf_synthetic.get_ball_scene_dataloader(
                n_scenes=dl_args['batch size'],
                n_balls=dl_args['number of objects'],
                max_radius=dl_args['max radius'])
        else:
            raise NotImplementedError
            
    elif dl_args["dataset"] == "cityscapes":
        return cityscapes.get_inr_loader_for_cityscapes(bsz=dl_args['batch size'],
            subset=dl_args['subset'], size=dl_args['image shape'], mode=dl_args['seg type'])
    
    elif dl_args["dataset"] == "inet12":
        N = dl_args.get('datapoints per class', None)
        return inet.get_inr_loader_for_inet12(bsz=dl_args['batch size'], subset=dl_args['subset'], N=N)
    
    else:
        raise NotImplementedError(dl_args["dataset"])

def get_val_inr_dataloader(dl_args):
    dl_args['subset'] = 'val'
    if dl_args["dataset"] == "inet12":
        dl_args['batch size'] = 192
    else:
        dl_args['batch size'] *= 4
    while True:
        dl = get_inr_dataloader(dl_args)
        for data in dl:
            yield data

def get_inr_loader_for_cifar10(ds_name, bsz, subset):
    paths = glob2(f"{DS_DIR}/dinet/{ds_name}_{subset}/*.pt")
    keys = siren.get_siren_keys()
    if len(paths) == 0:
        raise FileNotFoundError(f"{DS_DIR}/dinet/{ds_name}_{subset}/*.pt")
        
    def random_loader():
        inrs, classes = [], []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                data, subset_labels = torch.load(path)
                indices = list(range(len(data)))
                np.random.shuffle(indices)
                for ix in indices:
                    param_dict = {k:data[k][ix] for k in keys}
                    try:
                        inr = siren.Siren(out_channels=3)
                        inr.load_state_dict(param_dict)
                        inrs.append(inr)
                        classes.append(torch.tensor(subset_labels[ix]['cls']))
                    except RuntimeError:
                        continue
                    if len(inrs) == bsz:
                        inrbatch = siren.batchify(inrs).cuda()
                        yield inrbatch, torch.stack(classes).cuda()
                        inrs, classes = [], []
    return random_loader()


def get_inr_loader_for_imgds(ds_name, bsz, subset):
    paths = glob2(f"{DS_DIR}/dinet/{ds_name}/{subset}_*.pt")
    if len(paths) == 0:
        raise ValueError('bad dataloader specs')
    siren.get_siren_keys()
    def random_loader():
        inrs = []
        while True:
            np.random.shuffle(paths)
            for path in paths:
                inr = siren.Siren(out_channels=3)
                param_dict = torch.load(path)
                inr.load_state_dict(param_dict)
                inrs.append(inr)
                if len(inrs) == bsz:
                    yield siren.batchify(inrs).cuda()
                    inrs = []
    return random_loader()

