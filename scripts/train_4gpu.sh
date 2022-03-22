#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --num-gpus 4 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_4gpu.yaml

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 \
#	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 \
# 	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

#2>&1 &    
# | tee log/train_log_$ID.txt
