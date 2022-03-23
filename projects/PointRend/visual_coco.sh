export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml \
  --input '/cluster/work/cvl/leikel/hr_bound_project/detectron2-exp8-detach-semantic-light-details-nop1-postconv/sel_coco_images/*.jpg' \
  --output 'result_vis_pointrend_coco_selnew/' \
  --opts MODEL.WEIGHTS ./model_final_736f5a.pkl
  #--opts MODEL.WEIGHTS ./output/bmask_rcnn_r50_2x_cityscapes_post/model_final.pth
