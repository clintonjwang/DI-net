# Metadata from
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
import torch
import torchvision
from collections import namedtuple
from torchvision import transforms
from glob import glob
import numpy as np

from dinet.inrs import siren
from dinet import DS_DIR

nearest = transforms.InterpolationMode('nearest')

def get_inr_loader_for_cityscapes(bsz, subset, size, mode):
    if mode == 'coarse':
        paths = glob(f"{DS_DIR}/dinet/cityscapes/{subset}_*.pt")
    else:
        paths = glob(f"{DS_DIR}/dinet/cityscapes/fine_{subset}_*.pt")
    if subset == 'train':
        N = 2975
        loop = True
    elif subset == 'val':
        N = 500
        loop = False
    assert len(paths) == N, 'incomplete subset'

    if N % bsz != 0:
        print('warning: dropping last minibatch')
        N = (N // bsz) * bsz

    shrink = transforms.Resize(size, interpolation=nearest)

    def random_loader(loop=True):
        while True:
            np.random.shuffle(paths)
            for path_ix in range(0,N,bsz):
                inrs = [siren.Siren(out_channels=3) for _ in range(bsz)]
                segs = []
                for i in range(bsz):
                    param_dict, seg = torch.load(paths[path_ix+i])
                    try:
                        inrs[i].load_state_dict(param_dict)
                    except RuntimeError:
                        param_dict['net.4.weight'] = param_dict['net.4.weight'].tile(3,1)
                        param_dict['net.4.bias'] = param_dict['net.4.bias'].tile(3)
                        inrs[i].load_state_dict(param_dict)
                    segs.append(seg)
                segs = torch.stack(segs, dim=0).cuda()
                yield siren.batchify(inrs).cuda(), shrink(segs)
            if not loop:
                break
    
    return random_loader(loop=loop)


def replace_segs(subset):
    if subset == 'train':
        N = 2975
    elif subset == 'val':
        N = 500

    ds = torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split=subset, mode='coarse', target_type='semantic',
            target_transform=seg_transform)

    for i in range(N):
        path = f"{DS_DIR}/dinet/cityscapes/{subset}_{i}.pt"
        Fseg = ds[i][1].squeeze(0)
        param_dict, Cseg = torch.load(path)
        #if (Fseg & Cseg).sum() / (Fseg | Cseg).sum() > .3:
        torch.save((param_dict, Fseg), path)

def add_fine_segs(subset='train', N=2975):
    ds = torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split=subset, mode='fine', target_type='semantic',
            target_transform=seg_transform)
    for i in range(N):
        path = f"{DS_DIR}/dinet/cityscapes/{subset}_{i}.pt"
        Fseg = ds[i][1].squeeze(0)
        param_dict, _ = torch.load(path)
        new_path = f"{DS_DIR}/dinet/cityscapes/fine_{subset}_{i}.pt"
        torch.save((param_dict, Fseg), new_path)

def get_seg_frequencies():
    ds = torchvision.datasets.Cityscapes(DS_DIR+'/cityscapes',
            split='train', mode='coarse', target_type='semantic',
            target_transform=seg_transform)
    totals = torch.zeros(7, dtype=torch.long)
    for _,seg in ds:
        totals += seg.sum(dim=(0,2,3))
    torch.save(totals, DS_DIR+'/dinet/cityscapes/trainsegdist.pt')

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# id_to_cat = {l.id:l.catId for l in labels}
def seg_transform(seg):
    size = (256,512)
    tx = transforms.Compose((transforms.ToTensor(),
             transforms.Resize(size, interpolation=nearest)))
    seg = tx(seg)*255.
    return torch.stack((
            (seg > 6.5) * (seg < 10.5),
            (seg > 10.5) * (seg < 16.5),
            (seg > 16.5) * (seg < 20.5),
            (seg > 20.5) * (seg < 22.5),
            (seg > 22.5) * (seg < 23.5),
            (seg > 23.5) * (seg < 25.5),
            (seg > 25.5)), dim=1)

