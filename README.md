# Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding
[Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding](https://arxiv.org/abs/2203.15442), CVPR, 2022.

by Jiabo Ye, Junfeng Tian, Ming Yan, Xiaoshan Yang, Xuwu Wang, Ji Zhang, Liang He, Xin Lin
## Installation

### 1.Prepare the environment

``` python
python==3.8.10
pytorch==1.10.2
transformers==4.18.0
mmdet==2.11.0
mmcv-full==1.3.18
einops==0.4.1
icecream==2.1.2
numpy==1.22.3
scipy==1.8.0
ftfy==6.1.1
```

The above is a tested environment. Other version of these packages may also be fine.

We recommmand to install mmdet from the source codes inside this repository (```./models/swin_model```).

### 2.Dataset preparation
We follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md).

### 3.Checkpoint preparation
```
mkdir checkpoints
```
You can also set the ```--bert_model``` to ```bert-base-uncased``` to download bert checkpoints online or put ```bert-base-uncased``` into ```checkpoints/``` manually.

To train our model on refcoco/refcoco+/refcocog datasets, you need checkpoints trained on MSCOCO that the overlapping images of test set are excluded. We provide pretrained checkpoints on [Google Drive](https://drive.google.com/drive/folders/1GTi32iEfsJdYNtcHCUQIbhMdL5YFByVF?usp=sharing). For referit/flickr datasets, you can simply use the pretrained checkpoint from [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

## Training and Evaluation

### 1. Training

We present bash scripts for training  on referit.

For single-gpu training (not validated)
```
bash train_referit_single_gpu.sh
```

For multi-gpu training
```
bash train_referit_multi_gpu.sh
```

It's similar to train the model on the other datasets. Differents is that on RefCOCOg, we recommend to set ```--max_query_len 40```, on RefCOCO+ We recommend to set ```--lr_drop 120```.

### 2.Evaluation

For single-gpu evaluation
```
bash eval_referit_single_gpu.sh
```

For multi-gpu evaluation
```
bash eval_referit_multi_gpu.sh
```
## Citation
```
@article{ye2022shifting,
  title={Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding},
  author={Ye, Jiabo and Tian, Junfeng and Yan, Ming and Yang, Xiaoshan and Wang, Xuwu and Zhang, Ji and He, Liang and Lin, Xin},
  journal={arXiv preprint arXiv:2203.15442},
  year={2022}
}
```
## Acknowledge
This codebase is partially based on [TransVG](https://github.com/djiajunustc/TransVG) and [Swin-Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).
