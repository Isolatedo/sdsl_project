CUDA_VISIBLE_DEVICES=3,4,5
nohup python train.py --backbone resnet --lr 0.001 --workers 12 --epochs 350 --batch-size 12 --gpu-ids 0,1,2 --checkname sdsl --eval-interval 1 --dataset jcai_region --loss-type ce > train_2_5.log 2>&1 &
