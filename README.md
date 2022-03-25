# Mask Transfiner
Mask Transfiner for High-Quality Instance Segmentation [Mask Transfiner, CVPR 2022]

This is the official pytorch implementation of [Transfiner](https://arxiv.org/abs/2111.13673) built on the open-source detectron2 **[Under Construction]**.

> [**Mask Transfiner for High-Quality Instance Segmentation**](https://arxiv.org/abs/2111.13673)           
> Lei Ke, Martin Danelljan, Xia Li, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu  
> CVPR, 2022

Highlights
-----------------
- **Transfiner:** High-quality instance segmentation with state-of-the-art performance and extreme details.
- **Novelty:** An efficient transformer targeting for high-resolution instance masks predictions based on the quadtree structure.
- **Efficacy:** Large mask and boundary AP improvements on three instance segmentation benchmakrs, including COCO, Cityscapes and BDD100k. 
- **Simple:** Small additional computation burden and easy to use.

<img src="figures/mask_transfiner_banner.gif" width="800">

<!-- <table>
    <tr>
          <td><center><img src="figures/fig_vis1_new.png" height="260">
<img src="figures/mask_transfiner_banner.gif" height="430">
              
Qualitative instance segmentation results of our transfiner, using ResNet-101-FPN and FCOS detector.
          </center></td>
</tr>
</table> -->

## Mask Transfiner with Quadtree Transformer
<img src="figures/transfiner-banner.png" width="800">


Results on COCO test-dev
------------
(Check Table 9 of the paper for full results, all methods are trained on COCO train2017. This is a reimplementation. Thus, the numbers might be slightly different from the ones reported in our original paper.)

| Backbone  | Method | mAP(mask) | 
|----------|--------|-----------|
Res-R50-FPN | Mask R-CNN (ICCV'17) | 34.2 |
Res-R50-FPN | PANet (CVPR'18) | 36.6 |
Res-R50-FPN | MS R-CNN (CVPR'19) | 35.6 |
Res-R50-FPN | PointRend (1x CVPR'20) | 36.3 |
Res-R50-FPN | BCNet (CVPR'21) | 38.4 | 
Res-R50-FPN | Transfiner (CVPR'22)  | 39.4 |
**Res-R50-FPN-DCN** | **Transfiner (CVPR'22)**  | **40.5** |

| Backbone  | Method | mAP(mask) |
|----------|--------|-----------|
Res-R101-FPN | Mask R-CNN (ICCV'17) | 36.1 | 
Res-R101-FPN | MS R-CNN (CVPR'19) | 38.3 |
Res-R101-FPN | BMask R-CNN (ECCV'20) | 37.7 | 
Res-R101-FPN | SOLOv2 (NeurIPS'20) | 39.7 | 
Res-R101-FPN | BCNet (CVPR'21) | 39.8|
Res-R101-FPN | Transfiner (CVPR'22) | 40.7, [Pretrained Model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/ETnkg_whugxDtty354f_RUYBlSOnb84HGyxJm6ZvsCsz3A?e=WWk3g1) | 
**Res-R101-FPN-DCN** | **Transfiner (CVPR'22)** | **42.2**, [Pretrained Model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/ETzpdd3_QwNMnUFQ3DeqFqMBIn08xZIpyxYFczX_jq_xEw?e=1TnYKj) | 

Introduction
-----------------
Two-stage and query-based instance segmentation methods have achieved remarkable results. However, their segmented masks are still very coarse. In this paper, we present Mask Transfiner for high-quality and efficient instance segmentation. Instead of operating on regular dense tensors, our Mask Transfiner decomposes and represents the image regions as a quadtree. Our transformer-based approach only processes detected error-prone tree nodes and self-corrects their errors in parallel. While these sparse pixels only constitute a small proportion of the total number, they are critical to the final mask quality. This allows Mask Transfiner to predict highly accurate instance masks, at a low computational cost. Extensive experiments demonstrate that Mask Transfiner outperforms current instance segmentation methods on three popular benchmarks, significantly improving both two-stage and query-based frameworks by a large margin of +3.0 mask AP on COCO and BDD100K, and +6.6 boundary AP on Cityscapes. 

<!-- <center>
<table>
    <tr>
          <td><center><img src="figures/framework_new.png" height="430"></center></td>
    </tr>
</table>
A brief comparison of mask head architectures, see our paper for full details.
<table>	
    <tr>
          <td><center><img src="figures/netcompare.png" height="270"></center></td>
    </tr>
</table>
</center> -->

## Step-by-step Installation
```
conda create -n transfiner python=3.7 -y
source activate transfiner
 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
 
# Coco api and visualization dependencies
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
# Boundary dependency
pip install scikit-image
pip install kornia==0.5.11
 
export INSTALL_DIR=$PWD
 
# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
 
# install transfiner
cd $INSTALL_DIR
git clone https://github.com/lkeab/transfiner.git
cd transfiner/
python3 setup.py build develop
 
unset INSTALL_DIR
```


## Dataset Preparation
Prepare for [coco2017](http://cocodataset.org/#home) dataset and [Cityscapes](https://www.cityscapes-dataset.com) following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets).

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

Multi-GPU Training and Evaluation on Validation set
---------------
```
bash scripts/train_4gpu_transfiner_3x_101.sh
```
Or
```
bash scripts/train_4gpu.sh
```

Pretrained Models
---------------
Download: [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lkeab_connect_ust_hk/EqYIfZDfFDhIrTcTpNP79ccBoZ6L1FxAXqQxtGiH4Q0Z0A?e=9buMwd)
```
  mkdir pretrained_models
  #And put the downloaded pretrained models in this directory.
```

Initial Weights
---------------
Download: [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/lkeab_connect_ust_hk/Evb1jo2xDKxJoGSJxJ0aej4B8dfRsR9F7KByvlyF4SOL_A?e=hT81c9)
```
  mkdir init_weights
  #And put the downloaded init models weights in this directory.
```

Testing on Test-dev
---------------
```
bash scripts/test_3x_transfiner_101_deform.sh
```

Visualization
---------------
```
bash scripts/visual.sh
```

Citation
---------------
If you find Mask Transfiner useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{transfiner,
    author={Ke, Lei and Danelljan, Martin and Li, Xia and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    title={Mask Transfiner for High-Quality Instance Segmentation},
    booktitle = {CVPR},
    year = {2022}
}  
```
