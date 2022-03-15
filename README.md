# Mask Transfiner
Mask Transfiner for High-Quality Instance Segmentation [Mask Transfiner, CVPR 2022]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official pytorch implementation of [Transfiner](https://arxiv.org/abs/2111.13673) built on the open-source detectron2.

> [**Mask Transfiner for High-Quality Instance Segmentation**](https://arxiv.org/abs/2111.13673)           
> Lei Ke, Martin Danelljan, Xia Li, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu  
> CVPR 2022

Highlights
-----------------
- **Transfiner:** High-quality instance segmentation with state-of-the-art performance and extreme details.
- **Novelty:** An efficient transformer targeting for high-resolution instance masks output based on the quadtree structure.
- **Efficacy:** Large mask and boundary AP improvements on three large benchmakrs, including COCO, Cityscapes and BDD100k. 
- **Simple:** Small additional computation burden and easy to use.

Visualization of Occluded Objects
-----------------
<table>
    <tr>
        <td><center><img src="figures/fig_vis2_new.png" height="260">
            
Qualitative instance segmentation results of our BCNet, using ResNet-101-FPN and Faster R-CNN detector. The bottom row visualizes squared heatmap of **object contour and mask predictions** by the two GCN layers for the occluder and occludee in **the same ROI region** specified by the red bounding box, which also makes the final segmentation result of BCNet more explainable than previous methods. The heatmap visualization of GCN-1 in fourth column example shows that **BCNet handles multiple occluders with in the same RoI by grouping them together**. See our paper for more visual examples and comparisons.
          </center></td>
</tr>
</table>
<table>
    <tr>
          <td><center><img src="figures/fig_vis1_new.png" height="260">
              
Qualitative instance segmentation results of our BCNet, using ResNet-101-FPN and FCOS detector.
          </center></td>
</tr>
</table>

Results on COCO test-dev
------------
(Check Table 8 of the paper for full results, all methods are trained on COCO train2017)

Detector(Two-stage) | Backbone  | Method | mAP(mask) |
|--------|----------|--------|-----------|
Faster R-CNN| Res-R50-FPN | Mask R-CNN (ICCV'17) | 34.2 |
Faster R-CNN| Res-R50-FPN | PANet (CVPR'18) | 36.6 |
Faster R-CNN| Res-R50-FPN | MS R-CNN (CVPR'19) | 35.6 |
Faster R-CNN| Res-R50-FPN | PointRend (1x CVPR'20) | 36.3 |
**Faster R-CNN**| **Res-R50-FPN** | **BCNet (CVPR'21)** | [**38.4**](scores/stdout_r50_frcnn.txt) | 
Faster R-CNN| Res-R101-FPN | Mask R-CNN (ICCV'17) | 36.1 | 
Faster R-CNN| Res-R101-FPN | MS R-CNN (CVPR'19) | 38.3 |
Faster R-CNN| Res-R101-FPN | BMask R-CNN (ECCV'20) | 37.7 | 
**Box-free** | Res-R101-FPN | SOLOv2 (NeurIPS'20) | 39.7 | 
**Faster R-CNN**|**Res-R101-FPN** | **BCNet (CVPR'21)** | [**39.8**](scores/stdout_frcnn.txt)|

Detector(One-stage) | Backbone | Method | mAP(mask) |
|--------|----------|--------|-----------|
FCOS| Res-R101-FPN | BlendMask (CVPR'20) | 38.4 | 
FCOS| Res-R101-FPN | CenterMask (CVPR'20) | 38.3 | 
FCOS| Res-R101-FPN | SipMask (ECCV'20) | 37.8 |
FCOS| Res-R101-FPN | CondInst (ECCV'20) | 39.1 |
**FCOS**| Res-R101-FPN | **BCNet (CVPR'21)**| [**39.6**](scores/stdout_fcos.txt), [Pretrained Model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EfiDFLLEawFJpruwuOl3h3ABBjAKysTf0qJQU80iaKbqYg?e=igzC51), [Submission File](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EVgMSMFwOmVDjAIB3LFusAMBTyTY-N_6qWbAWEBq_PK9xQ?e=5Lrmv7)|
FCOS|Res-X101 FPN| BCNet (CVPR'21) | [41.2](scores/stdout_fcos_x101.txt) |

Introduction
-----------------
Segmenting highly-overlapping objects is challenging, because typically no distinction is made between real object contours and occlusion boundaries. Unlike previous two-stage instance segmentation methods, **BCNet** models image formation as composition of two overlapping image layers, where the top GCN layer detects the occluding objects (occluder) and the bottom GCN layer infers partially occluded instance (occludee). **The explicit modeling of occlusion relationship with bilayer structure naturally decouples the boundaries of both the occluding and occluded instances, and considers the interaction between them during mask regression.** We validate the efficacy of bilayer decoupling on both one-stage and two-stage object detectors with different backbones and network layer choices. The network of BCNet is as follows:
<center>
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
</center>

## Step-by-step Installation
```
conda create -n bcnet python=3.7 -y
source activate bcnet
 
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
 
# FCOS and coco api and visualization dependencies
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
# Boundary dependency
pip install scikit-image
 
export INSTALL_DIR=$PWD
 
# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
 
# install BCNet
cd $INSTALL_DIR
git clone https://github.com/lkeab/BCNet.git
cd BCNet/
python3 setup.py build develop
 
unset INSTALL_DIR
```


## Dataset Preparation
Prepare for [coco2017](http://cocodataset.org/#home) dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets). And use our [converted mask annotations](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EW2ZVyev7e5Pr1fVfF2nn18BRod82j_jW5Z4ywYd1evq8Q?e=qj0Bbm) to replace original annotation file for bilayer decoupling training.

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

Multi-GPU Training and evaluation on Validation set
---------------
```
bash all.sh
```
Or
```
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 \
	--config-file configs/fcos/fcos_imprv_R_50_FPN.yaml 2>&1 | tee log/train_log.txt
```

Pretrained Models
---------------
FCOS-version download: [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EfiDFLLEawFJpruwuOl3h3ABBjAKysTf0qJQU80iaKbqYg?e=igzC51)
```
  mkdir pretrained_models
  #And put the downloaded pretrained models in this directory.
```

Testing on Test-dev
---------------
```
export PYTHONPATH=$PYTHONPATH:`pwd`
CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 \
	--config-file configs/fcos/fcos_imprv_R_101_FPN.yaml \
	--eval-only MODEL.WEIGHTS ./pretrained_models/xxx.pth 2>&1 | tee log/test_log.txt
```

Visualization
---------------
```
bash visualize.sh
```

Citation
---------------
If you find Mask Transfiner useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{ke2021bcnet,
    author={Ke, Lei and Danelljan, Martin and Li, Xia and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    title={Mask Transfiner for High-Quality Instance Segmentation},
    booktitle = {CVPR},
    year = {2021}
}  
```
