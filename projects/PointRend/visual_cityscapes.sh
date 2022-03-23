export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_cityscapes.yaml \
  --input '/cluster/work/cvl/leikel/hr_bound_project/BMaskR-CNN-ori/BMaskR-CNN/projects/BMaskR-CNN/cityscapes_sel_new/*.png' \
  --output 'result_vis_pointrend_city_new_selected_1124_new/' \
  --opts MODEL.WEIGHTS ./model_final_115bfb.pkl
  #--opts MODEL.WEIGHTS ./output/bmask_rcnn_r50_2x_cityscapes_post/model_final.pth
  #--input '/cluster/work/cvl/leikel/hr_bound_project/detectron2/images_to_show_cityscapes/*.png' \
  #--input '/cluster/work/cvl/leikel/hr_bound_project/BMaskR-CNN-ori/BMaskR-CNN/projects/BMaskR-CNN/cityscapes_sel/*frankfurt_000000_005898_leftImg8bit*.png' \
