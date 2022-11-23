"""INR classification"""
import os
import time
import torch
import wandb

from dinet.utils import jobs as job_mgmt, util
osp = os.path
nn = torch.nn
F = nn.functional

from dinet import inn, RESULTS_DIR
from dinet.data import dataloader
from dinet.inn import point_set
import dinet.baselines.classifier
import dinet.inn.nets.classifier
import dinet.baselines.hypernets
import dinet.baselines.nuft

def train_classifier(args: dict) -> None:
    def forward():
        if util.is_model_dinet(args) or args["network"]['type'].lower().startswith('mlp') or args["network"]['type'].lower().startswith('nuft'):
            logits = model(img_inr)
        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)
        return logits

    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    val_data_loader = dataloader.get_val_inr_dataloader(dl_args)
    global_step = 0
    loss_fxn = nn.CrossEntropyLoss()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
    top3 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()

    model = get_model(args)
    wandb.watch(model, log="all", log_freq=100)
    optimizer = util.get_optimizer(model, args)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=2,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(paths["job output dir"]),
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    ) as profiler:
        start_time = time.time()
        for img_inr, labels in data_loader:
            model.train()
            global_step += 1
            logits = forward()
            loss = loss_fxn(logits, labels)
            pred_cls = logits.topk(k=3).indices
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss':loss.item(),
                'train_t3_acc': top3(pred_cls, labels).item(),
                'train_t1_acc': top1(pred_cls, labels).item(),
                'mins_elapsed': (time.time() - start_time)/60,
            })
            print('.', end='', flush=True)
            
            if global_step % 200 == 0:
                torch.save(model.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            if global_step % 10 == 0:
                model.eval()
                img_inr, labels = next(val_data_loader)
                with torch.no_grad():
                    forward()
                    loss = loss_fxn(logits, labels)
                    pred_cls = logits.topk(k=3).indices
                
                wandb.log({'val_loss':loss.item(),
                    'val_t3_acc': top3(pred_cls, labels).item(),
                    'val_t1_acc': top1(pred_cls, labels).item(),
                })
                
                if global_step == 10: # only log once
                    img = img_inr.produce_images(*dl_args['image shape'])
                    wandb.log({f"img_val_{labels[0].item()}" : wandb.Image(img[0].detach().cpu().permute(1,2,0).numpy()),
                    f"img_val_{labels[1].item()}" : wandb.Image(img[1].detach().cpu().permute(1,2,0).numpy()),
                    f"img_val_{labels[2].item()}" : wandb.Image(img[2].detach().cpu().permute(1,2,0).numpy())})
            
            if global_step == 1: # only log once
                img = img_inr.produce_images(*dl_args['image shape'])
                wandb.log({f"img_train_{labels[0].item()}" : wandb.Image(img[0].detach().cpu().permute(1,2,0).numpy()),
                f"img_train_{labels[1].item()}" : wandb.Image(img[1].detach().cpu().permute(1,2,0).numpy()),
                f"img_train_{labels[2].item()}" : wandb.Image(img[2].detach().cpu().permute(1,2,0).numpy())})
            
            if global_step >= args["optimizer"]["max steps"]:
                break

            profiler.step()

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))

def test_inr_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1, N = 0,0,0
    origin = args['target_job']
    if "discretization type" in args["data loading"]:
        kwargs = {"data loading": args["data loading"]}
    model = load_model_from_job(origin, **kwargs).eval()
    orig_args = job_mgmt.get_job_args(origin)
    ntype = orig_args["network"]['type'].lower()

    with torch.no_grad():
        model.eval()
        for img_inr, labels in data_loader:
            if util.is_model_dinet(orig_args) or ntype.startswith('mlp') or ntype.startswith('nuft'):
                logits = model(img_inr)
            else:
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)

            pred_cls = logits.topk(k=3).indices
            top3 += (labels.unsqueeze(1) == pred_cls).amax(1).long().sum().item()
            top1 += (labels == pred_cls[:,0]).long().sum().item()
            N += labels.shape[0]

    open(osp.join(paths["job output dir"], "stats.txt"), 'w').write(f"{top3}, {top1}, {N}")

def get_model(args):
    ntype = args["network"]["type"]
    if ntype.lower().startswith('mlp'):
        return dinet.baselines.hypernets.MLPCls(args['data loading']['classes']).cuda()
    in_ch = args['data loading']['input channels']
    n_classes = args['data loading']['classes']
    kwargs = dict(in_channels=in_ch, out_dims=n_classes)
    
    if hasattr(dinet.baselines.classifier, ntype):
        module = getattr(dinet.baselines.classifier, ntype)
        model = module(**kwargs)
    elif hasattr(dinet.inn.nets.classifier, ntype):
        kwargs["sampler"] = point_set.get_sampler_from_args(args['data loading'])
        kwargs = {**kwargs, **args["network"]['conv']}
        module = getattr(dinet.inn.nets.classifier, ntype)
        model = module(**kwargs)
    elif ntype.lower().startswith('nuft'):
        kwargs["sampler"] = point_set.get_sampler_from_args(args['data loading'])
        model = dinet.baselines.nuft.NUFTCls(grid_size=(32,32), **kwargs)
    elif ntype.startswith("Tx"):
        sampler = point_set.get_sampler_from_args(args['data loading'])
        module = getattr(dinet.baselines.classifier, ntype[2:])
        base = module(**kwargs)
        img_shape = args["data loading"]["image shape"]
        model, _ = inn.conversion.translate_discrete_model(base.layers, img_shape, sampler=sampler)
        inn.dinet.replace_conv_kernels(model, k_type='mlp', k_ratio=args["network"]["kernel expansion ratio"])
    else:
        raise NotImplementedError(f"Network type {ntype} not implemented")
        
    return model.cuda()


def load_model_from_job(origin, **kwargs):
    orig_args = {**job_mgmt.get_job_args(origin), **kwargs}
    path = osp.expanduser(f"{RESULTS_DIR}/{origin}/weights/best.pth")
    model = get_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model
    