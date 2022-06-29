export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo_swinb.py --config-file configs/transfiner/mask_rcnn_swinb_FPN_3x.yaml \
  --input 'demo/sample_imgs/*.jpg' \
  --output 'vis_coco_swinb_sample_swin/' \
  --opts MODEL.WEIGHTS ./pretrained_model/transfiner_swinb_3x.pth
