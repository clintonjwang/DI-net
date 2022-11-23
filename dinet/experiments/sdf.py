from dataclasses import fields
import gc
import time
import wandb
import torch
from dinet import inn
from dinet.inn.fields import DiscretizedField
import dinet.baselines.seg
from dinet.data import dataloader
from dinet.inn import point_set
from dinet.utils import util
from dinet.inn import fields
from dinet.inn.nets import field2field


def load_model(args):
    ntype = args["network"]["type"]
    kwargs = dict(in_channels=4, out_channels=1)
    if hasattr(field2field, ntype):
        module = getattr(field2field, ntype)
        model = module(**kwargs)
    elif hasattr(dinet.baselines.seg, ntype):
        module = getattr(dinet.baselines.seg, ntype)
        model = module(**kwargs)
    elif ntype.startswith("Tx"):
        module = getattr(dinet.baselines.seg, ntype[2:])
        base = module(**kwargs)
        img_shape = args["data loading"]["image shape"]
        model, _ = inn.conversion.translate_discrete_model(base.layers, img_shape,
                                                           extrema=((-1, 1), (-1, 1), (-1, 1)))
        # if args["data loading"]["discretization type"] != "grid":
        # inn.dinet.replace_conv_kernels(model, k_type='mlp', k_ratio=args["network"]["kernel expansion ratio"])
        # if net_args['frozen'] is True:
        #     inn.dinet.freeze_layer_types(InrNet)
    else:
        raise NotImplementedError(f"Network type {ntype} not implemented")
    wandb.watch(model, log="all", log_freq=100)
    return model.cuda()

def train_nerf_to_sdf(args: dict) -> None:
    dl_args = args["data loading"]
    global_step = 0
    data_loader = dataloader.get_inr_dataloader(dl_args)
    discretizations = point_set.get_discretizations_for_args(args)
    in_disc = discretizations['input']
    out_disc = discretizations['output']
    test_in_disc = discretizations['test_in']
    test_out_disc = discretizations['test_out']

    model = load_model(args).cuda()
    optimizer = util.get_optimizer(model, args)
    start_time = time.time()
    for rgba_sdf in data_loader:
        global_step += 1
        # (B,N,4), (B,N,1)
        rgba, sdf_gt = rgba_sdf(in_disc.coords, out_disc.coords)
        if util.is_model_adaptive_dinet(args):
            def rgba_fxn(coords):
                return rgba_sdf(coords)[0]
            sdf_pred = model(rgba_fxn, in_disc.coords, out_disc.coords).values
            loss = ((sdf_gt - sdf_pred)**2).mean()
        elif util.is_model_dinet(args):
            rgba_field = DiscretizedField(in_disc, values=rgba)
            sdf_pred = model(rgba_field, out_disc.coords).values
            loss = ((sdf_gt - sdf_pred)**2).mean()
        else:
            voxels = util.BNc_to_Bcdims(rgba, in_disc.shape)
            sdf_pred = model(voxels)
            loss = ((util.BNc_to_Bcdims(sdf_gt, out_disc.shape) - sdf_pred)**2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log({
            "sdf_mse": loss.item(),
            'mins_elapsed': (time.time() - start_time)/60,
        })
        print('.', end='')

        test_dtype = test_out_disc.type
        if global_step % 20 == 0:
            del rgba, sdf_gt, sdf_pred, loss
            gc.collect()
            torch.cuda.empty_cache()

            rgba, _ = rgba_sdf(test_in_disc.coords)
            rgba_dense, sdf_gt = rgba_sdf(test_out_disc.coords)
            with torch.no_grad():
                if util.is_model_adaptive_dinet(args):
                    sdf_pred = model(rgba_fxn, test_in_disc.coords,
                                     test_out_disc.coords).values
                    loss = ((sdf_gt - sdf_pred)**2).mean()
                elif util.is_model_dinet(args):
                    rgba_field = DiscretizedField(test_in_disc, values=rgba)
                    sdf_pred = model(rgba_field, test_out_disc.coords).values
                    loss = ((sdf_gt - sdf_pred)**2).mean()
                else:
                    voxels = util.BNc_to_Bcdims(rgba, test_in_disc.shape)
                    sdf_pred = model(voxels)
                    loss = ((util.BNc_to_Bcdims(sdf_gt, test_out_disc.shape) - sdf_pred)**2).mean()
            wandb.log({"val_sdf_mse": loss.item()})

            if test_dtype.startswith("grid"):
                if util.is_model_dinet(args):
                    rgba_dense = fields.reorder_grid_data(
                        rgba_dense, test_out_disc)
                    sdf_pred = fields.reorder_grid_data(sdf_pred, test_out_disc)
                    sdf_gt = fields.reorder_grid_data(sdf_gt, test_out_disc)
                shape = test_out_disc.shape
                rgba_dense = util.BNc_to_npy(rgba_dense[:1], shape)
                sdf_pred = util.BNc_to_npy(sdf_pred[:1], shape)
                sdf_gt = util.BNc_to_npy(sdf_gt[:1], shape)
                if test_dtype == "grid":
                    rgba_dense = rgba_dense[:, :, shape[2]//2, :3]
                    sdf_pred = sdf_pred[:, :, shape[2]//2]
                    sdf_gt = sdf_gt[:, :, shape[2]//2]
                elif test_dtype == "grid_slice":
                    rgba_dense = rgba_dense[..., :3]

                wandb.log({
                    'rgba': wandb.Image(rgba_dense),
                    'sdf_pred': wandb.Image(sdf_pred),
                    'sdf_gt': wandb.Image(sdf_gt),
                })

        if global_step >= args["optimizer"]["max steps"]:
            break
