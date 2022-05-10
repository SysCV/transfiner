#python3 setup.py build develop #--no-deps
#python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 --dist-url tcp://0.0.0.0:12346 \
	--config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
        --eval-only MODEL.WEIGHTS ./pretrained_model/output_3x_transfiner_r101_deform.pth


