#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --num-gpus 4 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x_4gpu_transfiner.yaml

