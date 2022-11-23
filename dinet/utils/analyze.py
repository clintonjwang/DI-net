"""Analyze INR-Net behaviors"""
import gc
import os
import torch
osp = os.path
nn = torch.nn
F = nn.functional
import numpy as np

from dinet import inn, TMP_DIR
from dinet.data import inet
from dinet.experiments.classify import load_model_from_job

def analyze_interpolate_grid_to_qmc():
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        logit_fxn = model(img_inr).cuda()
        qmc_coords = logit_fxn.generate_discretization(sample_size=np.prod(img_shape), method='qmc')
        logit_fxn.change_discretization('grid')
        grid_coords = logit_fxn.generate_discretization(dims=img_shape)
        grid_logits = logit_fxn.eval()(grid_coords).cpu()
        grid_mask = (front_conv.mask_tracker, back_conv.mask_tracker)

        C = cdist(grid_coords.cpu().numpy(), qmc_coords.cpu().numpy())
        _, assigment = linear_sum_assignment(C)
        qmc_coords = qmc_coords[assigment]

        intermediate_logits = []
        intermediate_masks = []
        for alpha in np.arange(.05,1,.05):
            logit_fxn = model(img_inr).cuda()
            coords = alpha*qmc_coords + (1-alpha)*grid_coords
            intermediate_logits.append(logit_fxn.eval()(coords).cpu())
            intermediate_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        logit_fxn = model(img_inr).cuda()
        qmc_logits = logit_fxn.eval()(qmc_coords).cpu()
        qmc_mask = (front_conv.mask_tracker, back_conv.mask_tracker)

        torch.save((base_logits, grid_logits, grid_mask, qmc_logits, qmc_mask, intermediate_logits, intermediate_masks),
            osp.expanduser(f'{TMP_DIR}/analyze_logit_mismatch.pt'))




def analyze_output_variance_rqmc():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        logits = []
        masks = []
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for _ in range(20):
            logit_fxn = model(img_inr).cuda()
            coords = logit_fxn.generate_discretization(sample_size=np.prod(img_shape), method='rqmc')
            logits.append(logit_fxn.eval()(coords).cpu())
            masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        torch.save((base_logits, logits, masks), osp.expanduser(f'{TMP_DIR}/output_variance_rqmc.pt'))


def analyze_change_resolution_grid_vs_qmc():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break

    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        RES = np.round(np.logspace(4, 8, num=9, base=2)).astype(int)
        qmc_logits, grid_logits = [],[]
        qmc_masks, grid_masks = [],[]
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for res in RES:
            img_shape = (res,res)
            # logit_fxn = model(img_inr).cuda()
            # coords = logit_fxn.generate_discretization(sample_size=np.prod(img_shape), method='qmc')
            # qmc_logits.append(logit_fxn.eval()(coords).cpu())
            # qmc_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

            logit_fxn = model(img_inr).cuda()
            grid_coords = logit_fxn.generate_discretization(dims=img_shape, method='grid')
            grid_logits.append(logit_fxn.eval()(grid_coords).cpu())
            grid_masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

            # torch.save((base_logits, grid_logits, grid_masks),
            #     osp.expanduser(f'{TMP_DIR}/change_resolution_grid_only.pt'))
            torch.save((base_logits, grid_logits, grid_masks, qmc_logits, qmc_masks),
                osp.expanduser(f'{TMP_DIR}/change_resolution_grid_vs_qmc.pt'))
            gc.collect()
            torch.cuda.empty_cache()


def analyze_divergence_over_depth():
    img_shape = (32,32)
    data_loader = inet.get_inr_loader_for_inet12(1, 'test')
    for img_inr, label in data_loader:
        if label == 1:
            break
    base = load_model_from_job('inet_nn_train').cuda().eval()
    with torch.no_grad():
        img = img_inr.produce_images(*img_shape)
        base_logits = base(img).cpu()

        logits = []
        masks = []
        model, _ = inn.conversion.translate_discrete_model(base, img_shape)
        front_conv = model[0][0][0]
        back_conv = model[-2][-1][-1].sequential[1][0]
        front_conv.mask_tracker = None
        back_conv.mask_tracker = None
        for _ in range(20):
            logit_fxn = model(img_inr).cuda()
            coords = logit_fxn.generate_discretization(sample_size=np.prod(img_shape), method='rqmc')
            logits.append(logit_fxn.eval()(coords).cpu())
            masks.append((front_conv.mask_tracker, back_conv.mask_tracker))

        torch.save((base_logits, logits, masks),
            osp.expanduser(f'{TMP_DIR}/analyze_logit_mismatch.pt'))
