#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml 

