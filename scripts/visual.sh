export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/transfiner/mask_rcnn_R_50_FPN_3x.yaml \
  --input 'demo/sample_imgs/*.jpg' \
  --output 'vis_coco_r50_sample/' \
  --opts MODEL.WEIGHTS ./pretrained_model/output_3x_transfiner_r50.pth
