#python3 setup.py build develop #--no-deps
#python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

#CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --num-gpus 1 \
#	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 4 --dist-url tcp://0.0.0.0:12346 \
	--config-file configs/LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_finetune.yaml \
        --eval-only MODEL.WEIGHTS ./output_1x_finetune_lvis/model_final.pth

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 \
# 	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

#2>&1 &    
# | tee log/train_log_$ID.txt
