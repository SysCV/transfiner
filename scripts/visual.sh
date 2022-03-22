export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x_4gpu_transfiner_deform.yaml \
  --input 'datasets/coco/val2017/*.jpg' \
  --output 'vis_coco_ours_new1/' \
  --opts MODEL.WEIGHTS ./pretrained_model/output_3x_finetune_r101_deform.pth
