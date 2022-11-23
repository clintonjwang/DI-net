import os, torch
import wandb

from dinet.utils import losses, util
from dinet.utils import jobs as job_mgmt
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from dinet.data import dataloader
from dinet import inn, RESULTS_DIR
from dinet.inn import point_set
import dinet.inn.nets.convnext
import dinet.baselines.convnext
import dinet.baselines.classifier
import dinet.baselines.hypernets
import dinet.baselines.nuft
import dinet.inn.nets.field2field

rescale_float = mtr.ScaleIntensity()
from dinet.data import DS_DIR

class_names=('void', 'ground', 'building', 'traffic', 'nature', 'sky', 'human', 'vehicle')
def get_model_for_args(args):
    ntype = args["network"]["type"]
    kwargs = dict(in_channels=3, out_channels=7)
    if ntype == "convnext":
        return dinet.baselines.convnext.mini_convnext().cuda()

    elif ntype == "inr-convnext":
        sampler = point_set.get_sampler_from_args(args['data loading'])
        return dinet.inn.nets.convnext.translate_convnext_model(
                args["data loading"]["image shape"], sampler=sampler).cuda()

    elif ntype == "inr-mlpconv":
        sampler = point_set.get_sampler_from_args(args['data loading'])
        diNet = dinet.inn.nets.convnext.translate_convnext_model(
                args["data loading"]["image shape"], sampler=sampler)
        return inn.dinet.replace_conv_kernels(diNet, k_type='mlp').cuda()

    elif hasattr(inn.nets.field2field, ntype):
        module = getattr(inn.nets.field2field, ntype)
        kwargs["sampler"] = point_set.get_sampler_from_args(args['data loading'])

    elif hasattr(dinet.baselines.classifier, ntype):
        module = getattr(dinet.baselines.classifier, ntype)

    elif ntype.lower().startswith('nuft'):
        kwargs["grid_size"] = args["data loading"]["image shape"]
        kwargs["sampler"] = point_set.get_sampler_from_args(args['data loading'])
        module = dinet.baselines.nuft.NUFTSeg

    elif ntype.lower().startswith('hyper'):
        model = dinet.baselines.hypernets.HypernetSeg(198915).cuda()#
        return model.cuda()
        
    else:
        raise NotImplementedError(f"Network type {ntype} not implemented")
        
    model = module(**kwargs)
    return model.cuda()

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"{RESULTS_DIR}/{origin}/weights/model.pth")
    model = get_model_for_args(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def interpolate_gt_seg(seg, coords):
    coo = torch.floor(coords).long()
    return seg[...,coo[:,0], coo[:,1]].transpose(1,2)

class MaskedCELoss(nn.CrossEntropyLoss):
    def forward(self, logits, gt_labels, mask):
        return super().forward(logits[mask], gt_labels[mask])


def train_segmenter(args: dict) -> None:
    paths = args["paths"]
    dl_args = args["data loading"]
    global_step = 0
    trainsegdist = torch.load(DS_DIR+'/dinet/cityscapes/trainsegdist.pt')
    weight = trainsegdist.sum() / trainsegdist
    loss_fxn = MaskedCELoss(weight=weight.cuda())
    dims = dl_args['image shape']
    ntype = args["network"]["type"]
    discretizations = point_set.get_discretizations_for_args(args)
    in_disc = discretizations['input']
    out_disc = discretizations['output']
    test_in_disc = discretizations['test_in']
    test_out_disc = discretizations['test_out']
    if in_disc == 'masked':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)

    # columns=["id", "source", "warped", "target"]
    # predictions_table = wandb.Table(columns = columns)
    class_labels = {i:class_names[i] for i in range(len(class_names))}

    model = get_model_for_args(args)
    wandb.watch(model, log="all", log_freq=100)
    optimizer = util.get_optimizer(model, args)
    grid_outputs = dl_args['discretization'] == 'grid' or not util.is_model_dinet(args)
    for img_inr, segs in data_loader: # _, (B,n_cls,*dims)
        global_step += 1
        
        if util.is_model_dinet(args): # DI-Net
            logit_inr = model(img_inr) # (B,N,3)
            if grid_outputs:
                logit_inr.sort()
            logits = logit_inr.values
        elif ntype.lower().startswith('nuft'):
            logits = util.Bcdims_to_BNc(model(img_inr))
        elif ntype.lower().startswith('hyper'):
            logit_inr = model(img_inr) # (B,N,3)
            logits = logit_inr(in_disc.coords)
        else: # CNN
            img = img_inr.produce_images(*dims)
            logits = util.Bcdims_to_BNc(model(img))

        if grid_outputs:
            seg_gt = util.Bcdims_to_BNc(segs)
        else:
            seg_gt = interpolate_gt_seg(segs, out_disc.coords) #logit_inr.coords

        maxes, gt_labels = seg_gt.max(-1)
        mask = maxes != 0
        loss = loss_fxn(logits, gt_labels, mask=mask)
        pred_seg = logits.max(-1).indices
        pred_1hot = F.one_hot(pred_seg, num_classes=seg_gt.size(-1)).bool()
        iou = losses.mean_iou(pred_1hot[mask], seg_gt[mask]).item()
        acc = losses.pixel_acc(pred_1hot[mask], seg_gt[mask]).item()
        wandb.log({"train_loss": loss.item(), "train_mIoU": iou, "train_PixAcc": acc})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print('.', end='')

        # test_dtype = dl_args["test discretization"]["type"]
        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))

            # if test_dtype == "grid":
            with torch.no_grad():
                img = img_inr.produce_images(*dims)[0]
                gt_labels += 1
                gt_labels[~mask] = 0
                pred_seg += 1
                pred_seg[~mask] = 0
                
            pred_seg = pred_seg[0].reshape(*dims)
            gt_seg = gt_labels[0].reshape(*dims)
            img = util.rgb2d_tensor_to_npy(img)
            wandb.log({
                f"{global_step} predicted" : wandb.Image(img, masks={
                    "predictions" : {
                        "mask_data" : util.grayscale2d_tensor_to_npy(pred_seg),
                        "class_labels" : class_labels
                    },
                }),
                f"{global_step} GT" : wandb.Image(img, masks={
                    "ground_truth" : {
                        "mask_data" : util.grayscale2d_tensor_to_npy(gt_seg),
                        "class_labels" : class_labels
                    }
                }),
            })

        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))


    
def test_inr_segmenter(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    if dl_args['discretization type'] == 'masked':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)
    class_labels = {i:class_names[i] for i in range(len(class_names))}

    origin = args['target_job']
    model = load_model_from_job(origin).eval()
    orig_args = job_mgmt.get_job_args(origin)
    ix = 0
    if dl_args['discretization type'] == 'grid':
        dims = dl_args['image shape']
    grid_outputs = dl_args['discretization type'] == 'grid' or not util.is_model_dinet(orig_args)

    ntype = orig_args["network"]["type"]
    discretizations = point_set.get_discretizations_for_args(orig_args)
    in_disc = discretizations['test_in']
    out_disc = discretizations['test_out']
    val_iou, val_acc, N = 0, 0, 0
    with torch.no_grad():
        for img_inr, segs in data_loader:
            ix += 1
            if util.is_model_dinet(orig_args): # DI-Net
                logit_inr = model(img_inr) # (B,N,3)
                if grid_outputs:
                    logit_inr.sort()
                logits = logit_inr.values
            elif ntype.lower().startswith('nuft'):
                logits = util.Bcdims_to_BNc(model(img_inr))
            elif ntype.lower().startswith('hyper'):
                logit_inr = model(img_inr) # (B,N,3)
                logits = logit_inr(in_disc.coords)
            else: # CNN
                img = img_inr.produce_images(*dims)
                logits = util.Bcdims_to_BNc(model(img))

            if grid_outputs:
                seg_gt = util.Bcdims_to_BNc(segs)
            else:
                seg_gt = interpolate_gt_seg(segs, logit_inr.coords)

            maxes, gt_labels = seg_gt.max(-1)
            mask = maxes != 0
            pred_seg = logits.max(-1).indices
            pred_1hot = F.one_hot(pred_seg, num_classes=seg_gt.size(-1)).bool()
            iou = losses.mean_iou(pred_1hot[mask], seg_gt[mask]).item()
            acc = losses.pixel_acc(pred_1hot[mask], seg_gt[mask]).item()
            val_iou += iou
            val_acc += acc
            N += 1
            print('.', end='')

            # if ix == 0:
            #     img = img_inr.produce_images(*dims)[0]
            #     gt_labels += 1
            #     gt_labels[~mask] = 0
            #     pred_seg += 1
            #     pred_seg[~mask] = 0
                    
            #     pred_seg = pred_seg[0].reshape(*dims)
            #     gt_seg = gt_labels[0].reshape(*dims)
            #     img = util.rgb2d_tensor_to_npy(img)
            #     wandb.log({
            #         f"predicted" : wandb.Image(img, masks={
            #             "predictions" : {
            #                 "mask_data" : util.grayscale2d_tensor_to_npy(pred_seg),
            #                 "class_labels" : class_labels
            #             },
            #         }),
            #         f"GT" : wandb.Image(img, masks={
            #             "ground_truth" : {
            #                 "mask_data" : util.grayscale2d_tensor_to_npy(gt_seg),
            #                 "class_labels" : class_labels
            #             }
            #         }),
            #     })
    open(osp.join(paths["job output dir"], "stats.txt"), 'w').write(f"{val_iou/N}, {val_acc/N}")
 
import imgviz
def save_example_segs(path, rgb, pred_seg, gt_seg):
    label_names = [
        "{}:{}".format(i, n) for i, n in enumerate(class_names)
    ]
    labelviz_pred = imgviz.label2rgb(pred_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    labelviz_gt = imgviz.label2rgb(gt_seg.cpu())#, label_names=label_names, font_size=6, loc="rb")
    rgb = rescale_float(rgb.cpu().permute(1,2,0))

    # kwargs = dict(bbox_inches='tight', transparent="True", pad_inches=0)
    plt.figure(dpi=400)
    plt.tight_layout()
    plt.subplot(131)
    # plt.title("rgb")
    plt.imshow(rgb)
    plt.axis("off")
    plt.subplot(132)
    # plt.title("pred")
    plt.imshow(labelviz_pred)
    plt.axis("off")
    plt.subplot(133)
    # plt.title("gt")
    plt.imshow(labelviz_gt)
    plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)

    img = imgviz.io.pyplot_to_numpy()
    plt.imsave(path, img)
    plt.close('all')
