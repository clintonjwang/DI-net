# Discretization Invariance for Deep Learning on Neural Fields

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/clintonjwang/DI-Net/blob/main/LICENSE)

[Project](https://clintonjwang.github.io/di-net) | [Paper](https://arxiv.org/abs/2206.01178)

![DI-Nets learn directly from datasets of neural fields](https://github.com/clintonjwang/di-net/blob/main/teaser.png?raw=true)

**DI-Net** is a framework for learning discretization invariant operators on neural fields. This repository focuses on convolutional DI-Nets, continuous generalizations of CNNs, which can be trained to classify and segment visual data represented by neural fields under various discretizations.

## Reproducing our experiments

Start by setting paths in `dinet/__init__.py` according to where you want to store data and temp files.

### Creating datasets of neural fields

In our paper, we create neural fields by fitting the [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/) for segmentation and the [ImageNet1k dataset](https://image-net.org/download.php) for classification. The 12 labels for classifying ImageNet1k are derived from the [big_12 subset](https://robustness.readthedocs.io/en/latest/example_usage/custom_imagenet.html#basic-usage-loading-pre-packaged-imagenet-based-datasets).

After downloading the datasets, you need to modify `fit_inr.py` to specify where your images are located.
By default, each job only fits a subset of the dataset (`end_ix` in `configs/fit_config.yaml` specifies how many images are fit per job). Modify `fit_inr.py` to change this behavior. To neural fields, run the following command on a GPU for each subset:
`python fit_inr.py -d=[dataset_name] -s=[start_index]`
* `-d` must match one of the entries in `configs/fit_config.yaml`, i.e. `inet12` (ImageNet fit with SIREN), `inet_rff` (ImageNet fit with Random Fourier Features), or * `cityscapes` (SIREN)
* `-s` is the index of the dataset to start this job from, for fitting in parallel. Start with `0`, then should be `end_ix`, `2*end_ix`, etc.

### Training and evaluating DI-Net

To train a DI-Net, run `python train.py -j=[train_jobname] -c=[config_name]` where config_name must be one of the filenames in `configs/*/*.yaml`, e.g. `-c=inet_i2`.

`configs/` contains the following settings for `train.py`:
- `inet/`: ImageNet classification
  - `inet_i2`: train DI-Net-2 with QMC discretization
  - `inet_i2g`: train DI-Net-2 with grid discretization
  - `inet_i2s`: train DI-Net-2 with shrunk discretization
  - `inet_i4`: train DI-Net-4 with QMC
  - `inet_nn2`: train 2-layer CNN (32x32 resolution)
  - `inet_nn4`: train 4-layer CNN
  - `inet_mlp`: train MLP mapping SIREN params to class label
  - `inet_nuft`: train non-uniform CNN ([Jiang et al.](https://arxiv.org/abs/1901.02070)) 
  - `snet_train`: train truncated EfficientNet translated to DI-Net
  - `cnext_train`: train truncated ConvNext translated to DI-Net
- `seg/`: CityScapes segmentation
  - `seg_i3`: train DI-Net-3
  - `seg_i5`: train DI-Net-5
  - `seg_nn3`: train 3-layer fully convolutional network (FCN)
  - `seg_nn5`: train 5-layer FCN
  - `seg_hyper`: train hypernetwork on SIREN parameters
  - `seg_nuft`: train non-uniform CNN ([Jiang et al.](https://arxiv.org/abs/1901.02070))
- `sdf/`: signed distance function prediction
  - `sdf_i3`: train/test DI-Net with QMC discretization
  - `sdf_i3g`: train/test DI-Net with grid discretization
  - `sdf_i3r`: train/test DI-Net with random Monte Carlo discretization
  - `sdf_i3s`: train/test DI-Net with shrunk discretization
  - `sdf_i3qs`: train DI-Net with randomized QMC discretization and test with shrunk discretization
  - `sdf_i3as`: train DI-Net with mixed discretization and test with shrunk discretization
  - following the examples in `sdf_i3as` and `sdf_i3qs` you can train/test all other discretization combinations
  - `sdf_nn3`: train/test fully convolutional network


For signed distance function prediction, the validation statistics and examples are saved during training. For classification and segmentation, run `python infer.py -j=[val_jobname] -c=[config_name] --target_job=[train_jobname]` to perform validation.

`configs/` contains the following settings for `infer.py`:
- `inet/`: ImageNet classification
  - `inet_val`: assess model performance on validation images fit to RFF (default)
  - `ival_siren`: assess model performance on validation images fit to SIREN (for MLP)
  - `ival_half`: assess model on 16x16 resolution
  - `ival_1x5`: assess model on 48x48 resolution
  - `ival_2x`: assess model on 64x64 resolution
  - `ival_2x5`: assess model on 80x80 resolution
  - `ival_3x`: assess model on 96x96 resolution
  - `ival_q`: assess model on QMC discretization
  - `ival_s`: assess model on shrunk discretization
  - `ival_g`: assess model with grid discretization
- `seg/`: CityScapes segmentation
  - `seg_val_c`: assess model performance on coarse segmentations (in-distribution)
  - `seg_val`: assess model performance on fine segmentations (out-of-distribution)


Additional functions for analyzing the data can be found in `utils/analyze.py`.

## Requirements

CUDA is required. The code cannot run on CPU.
Most code was run on NVIDIA Quadro RTX 5000.

## Citation

**[Approximate Discretization Invariance for Deep Learning on Implicit Neural Datasets](https://arxiv.org/abs/2206.01178)**<br>
[Clinton J. Wang](https://clintonjwang.github.io/) and [Polina Golland](https://people.csail.mit.edu/polina/)<br>
[NeurIPS Workshop on Symmetry and Geometry in Neural Representations 2022](https://www.neurreps.org/)

If you find this work useful please use the following citation:
```
@misc{wang2022dinet,
      title={Approximate Discretization Invariance for Deep Learning on Implicit Neural Datasets}, 
      author={Clinton J. Wang and Polina Golland},
      year={2022},
      eprint={2206.01178},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

Thanks to [Neel Dey](https://www.neeldey.com/) and [Daniel Moyer](https://dcmoyer.github.io/) for their many helpful suggestions.
